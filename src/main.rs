use std::{cell::RefCell, os::unix::process::parent_id, rc::Rc};

use eframe::egui::{CentralPanel, Color32, DragValue, Frame, Pos2, Rect, Scene, Sense, SidePanel, Stroke, Ui, Vec2};
use rand::Rng;

fn gen_tree(resolution: f32) -> NodeRef {
    let mut tree = random_tree(0.0, None);

    let f = |x: f32| (x * 10.0).cos();
    let f = InputFunction::from_func(Rc::new(f));
    insert_function_rec(tree.clone(), resolution, 20, f.clone());
    tree
}

fn main() {
    let mut resolution: f32 = -7.0;
    let mut tree = gen_tree(resolution.exp());
    
    let mut sample_y = 1.0;

    let mut scene_rect = Rect::ZERO;
    eframe::run_simple_native("tree test", Default::default(), move |ctx, _frame| {
        SidePanel::left("leeft").show(ctx, |ui| {
            ui.label("Sample y: ");
            ui.add(DragValue::new(&mut sample_y).range(0.0..=1.0).speed(1e-2));

            ui.label("Resolution: ");
            let resp = ui.add(DragValue::new(&mut resolution).speed(1e-1));
            if resp.changed() {
                tree = gen_tree(dbg!(resolution.exp()));
                dbg!(tree.as_ptr());
            }
        });

        CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                Scene::new()
                    .zoom_range(0.0..=100.0)
                    .show(ui, &mut scene_rect, |ui| {
                        draw_tree(ui, tree.clone(), sample_y);
                    });
            });
        });
    })
    .unwrap();
}

type NodeRef = Rc<RefCell<Node>>;

#[derive(Clone)]
struct Node {
    parent: Option<NodeRef>,
    content: NodeContent,
    level: usize,
}

#[derive(Clone)]
enum NodeContent {
    Branch([NodeRef; 4]),
    Leaf(f32),
}

fn random_tree(p: f64, parent: Option<NodeRef>) -> NodeRef {
    let new_node = Rc::new(RefCell::new(Node {
        level: parent.as_ref().map(|p| p.borrow().level + 1).unwrap_or(0),
        parent,
        content: NodeContent::Leaf(rand::thread_rng().r#gen()),
    }));

    if rand::thread_rng().gen_bool(p.min(1.0)) {
        let content =
            NodeContent::Branch([(); 4].map(|_| random_tree(p / 4.0, Some(new_node.clone()))));
        new_node.borrow_mut().content = content;
    }

    new_node
}

fn draw_tree(ui: &mut Ui, root: NodeRef, sample_y: f32) {
    let (rect, resp) = ui.allocate_exact_size(Vec2::splat(500.0), Sense::click_and_drag());
    /*
    zero_leaves(&root);

    // Show hover and neighbors
    if let Some(interact) = resp.hover_pos() {
        if let Some(found) = find_node_recursive(interact, root.clone(), rectangle) {
            if let NodeContent::Leaf(value) = &mut found.borrow_mut().content {
                *value = 0.95;
            }

            for edge in [Edge::Top, Edge::Bottom, Edge::Left, Edge::Right] {
                //eprintln!("BEGIN NEIGHBOR {edge:?}");
                find_neighbors(&found, edge, &mut |node| {
                    //eprintln!("MUT");
                    //debug_borrow!(node);
                    if let NodeContent::Leaf(value) = &mut node.borrow_mut().content {
                        *value = 0.75;
                    }
                });
                //eprintln!("END NEIGHBOR");
            }

            /*
            if resp.clicked() || resp.dragged() {
                let parent = found.clone();
                found.borrow_mut().content = NodeContent::Branch([(); 4].map(|_| {
                    Rc::new(RefCell::new(Node {
                        parent: Some(parent.clone()),
                        content: NodeContent::Leaf(0.0),
                    }))
                }));
            }
            */
        }
    }
    */

    draw_tree_recursive(ui, &root, rect);

    ui.painter().line_segment([Pos2::new(0.0, sample_y), Pos2::new(1.0, sample_y)], Stroke::new(1.0, Color32::LIGHT_GRAY));
    draw_func_at_y(&root, ui, rect, sample_y, rect.max.y + 100.0, 90.0);
}

fn draw_tree_recursive(ui: &mut Ui, node: &NodeRef, rect: Rect) {
    match &node.borrow().content {
        NodeContent::Leaf(value) => {
            let fill = if *value > 0.0 {
                Color32::BLACK.lerp_to_gamma(Color32::ORANGE, *value)
            } else {
                Color32::BLACK.lerp_to_gamma(Color32::LIGHT_BLUE, -value)
            };
            //let fill = Color32::from_gray((255.0 * value) as u8);
            let stroke = Stroke::new(0.1, Color32::WHITE);
            ui.painter()
                .rect(rect, 0.0, fill, stroke, eframe::egui::StrokeKind::Outside);
        }
        NodeContent::Branch(branches) => {
            let (lefts, rights) = rect.split_left_right_at_fraction(0.5);
            let (top_left, bottom_left) = lefts.split_top_bottom_at_fraction(0.5);
            let (top_right, bottom_right) = rights.split_top_bottom_at_fraction(0.5);
            let rects = [top_left, top_right, bottom_left, bottom_right];

            for (branch, rect) in branches.iter().zip(rects) {
                draw_tree_recursive(ui, branch, rect);
            }
        }
    }
}

fn find_node_recursive(pos: Pos2, node: NodeRef, rect: Rect) -> Option<NodeRef> {
    match &node.borrow().content {
        NodeContent::Leaf(_) => rect.contains(pos).then(|| node.clone()),
        NodeContent::Branch(branches) => {
            let (lefts, rights) = rect.split_left_right_at_fraction(0.5);
            let (top_left, bottom_left) = lefts.split_top_bottom_at_fraction(0.5);
            let (top_right, bottom_right) = rights.split_top_bottom_at_fraction(0.5);
            let rects = [top_left, top_right, bottom_left, bottom_right];

            for (branch, rect) in branches.iter().zip(rects) {
                if rect.contains(pos) {
                    if let Some(found) = find_node_recursive(pos, branch.clone(), rect) {
                        return Some(found);
                    }
                }
            }

            None
        }
    }
}

fn zero_leaves(node: &NodeRef) {
    match &mut node.borrow_mut().content {
        NodeContent::Leaf(value) => *value = 0.0,
        NodeContent::Branch(branches) => {
            for branch in branches {
                zero_leaves(branch);
            }
        }
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
enum Quadrant {
    TopLeft = 0,
    TopRight = 1,
    BotLeft = 2,
    BotRight = 3,
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(usize)]
enum Edge {
    Top = 0,
    Bottom = 1,
    Left = 2,
    Right = 3,
}

impl Edge {
    fn neighbor_quadrants(&self) -> [Quadrant; 2] {
        match self {
            Edge::Top => [Quadrant::BotLeft, Quadrant::BotRight],
            Edge::Bottom => [Quadrant::TopLeft, Quadrant::TopRight],
            Edge::Left => [Quadrant::TopRight, Quadrant::BotRight],
            Edge::Right => [Quadrant::TopLeft, Quadrant::BotLeft],
        }
    }
}

impl Quadrant {
    fn mirror(&self, edge: Edge) -> Self {
        let horiz = matches!(edge, Edge::Left | Edge::Right);

        match (self, horiz) {
            (Self::TopLeft, false) => Self::BotLeft,
            (Self::TopRight, false) => Self::BotRight,
            (Self::BotLeft, false) => Self::TopLeft,
            (Self::BotRight, false) => Self::TopRight,

            (Self::TopLeft, true) => Self::TopRight,
            (Self::TopRight, true) => Self::TopLeft,
            (Self::BotLeft, true) => Self::BotRight,
            (Self::BotRight, true) => Self::BotLeft,
        }
    }
}
/*
impl Quadrant {
    /// At this quadrant, which edges are shared by this quadrant in the same branch?
    fn adjacent_edges(&self) -> [Edge; 2] {
        match self {
            Self::TopLeft => [Edge::Right, Edge::Bottom],
            Self::TopRight => [Edge::Left, Edge::Bottom],
            Self::BotLeft => [Edge::Right, Edge::Top],
            Self::BotRight => [Edge::Left, Edge::Top],
        }
    }
}
*/

fn adjacent_quadrant(quad: Quadrant, edge: Edge) -> Option<Quadrant> {
    match (quad, edge) {
        (Quadrant::TopLeft, Edge::Right) => Some(Quadrant::TopRight),
        (Quadrant::BotLeft, Edge::Right) => Some(Quadrant::BotRight),

        (Quadrant::TopRight, Edge::Left) => Some(Quadrant::TopLeft),
        (Quadrant::BotRight, Edge::Left) => Some(Quadrant::BotLeft),

        (Quadrant::TopLeft, Edge::Bottom) => Some(Quadrant::BotLeft),
        (Quadrant::TopRight, Edge::Bottom) => Some(Quadrant::BotRight),

        (Quadrant::BotLeft, Edge::Top) => Some(Quadrant::TopLeft),
        (Quadrant::BotRight, Edge::Top) => Some(Quadrant::TopRight),
        _ => None,
    }
}

/*
#[macro_export]
macro_rules! debug_borrow {
    ($var:ident) => {
        {
            let ptr = ::std::rc::Rc::as_ptr(&$var);
            eprintln!(
                "[{}:{}:{}] {}: {:?} borrow",
                file!(),
                line!(),
                column!(),
                stringify!($var),
                ptr
            );
        }
    };
}
*/

fn find_neighbors(node: &NodeRef, edge: Edge, callback: &mut impl FnMut(&NodeRef)) {
    if let Some(root_neighbor) = find_neighbor_up(node, edge) {
        find_neighbors_down(&root_neighbor, edge, callback);
    }
}

fn find_neighbor_up(node: &NodeRef, edge: Edge) -> Option<NodeRef> {
    let Some(parent) = node.borrow().parent.clone() else {
        // root has no neighbors
        return None;
    };
    let NodeContent::Branch(siblings) = &parent.borrow().content.clone() else {
        panic!("Invalid tree; parent node is a leaf!");
    };
    let quad = siblings
        .iter()
        .position(|branch| Rc::ptr_eq(&node, branch))
        .map(Quadrant::from)
        .unwrap();
    if let Some(adj) = adjacent_quadrant(quad, edge) {
        // if we have a sibling on the edge we're checking, that's our neighbor.
        Some(siblings[adj as usize].clone())
    } else {
        let parent_sibling = find_neighbor_up(&parent, edge)?;
        match &parent_sibling.borrow().content {
            // if we don't have a direct sibling, our neighbor is one of the children of our parent's neighbor.
            NodeContent::Branch(branches) => Some(branches[quad.mirror(edge) as usize].clone()),
            // if our parent's neighbor has no children, then that's our neighbor.
            NodeContent::Leaf(_) => Some(parent_sibling.clone()),
        }
    }
}

fn find_neighbors_down(node: &NodeRef, edge: Edge, callback: &mut impl FnMut(&NodeRef)) {
    let content = node.borrow().content.clone();
    match content {
        NodeContent::Leaf(_) => callback(&node),
        NodeContent::Branch(children) => {
            for quad in edge.neighbor_quadrants() {
                find_neighbors_down(&children[quad as usize], edge, callback);
            }
        }
    }
}

impl From<usize> for Quadrant {
    fn from(value: usize) -> Self {
        match value {
            0 => Quadrant::TopLeft,
            1 => Quadrant::TopRight,
            2 => Quadrant::BotLeft,
            3 => Quadrant::BotRight,
            _ => panic!("Incorrect quadrant index"),
        }
    }
}

type UserFunc = Rc<dyn Fn(f32) -> f32>;

#[derive(Clone)]
struct InputFunction {
    begin: f32,
    end: f32,
    f: UserFunc,
    level: usize,
}

impl InputFunction {
    fn from_func(f: UserFunc) -> Self {
        Self {
            begin: 0.0,
            end: 1.0,
            f,
            level: 0,
        }
    }

    fn right(&self) -> Self {
        let width = self.end - self.begin;
        Self {
            begin: self.begin + width / 2.0,
            end: self.end,
            f: self.f.clone(),
            level: self.level + 1,
        }
    }

    fn left(&self) -> Self {
        let width = self.end - self.begin;
        Self {
            begin: self.begin,
            end: self.end - width / 2.0,
            f: self.f.clone(),
            level: self.level + 1,
        }
    }

    fn call(&self, x: f32) -> f32 {
        (&self.f)(x)
    }
}

fn refine_cell(node: NodeRef, input_function: InputFunction) {
    let level = node.borrow().level;
    let NodeContent::Leaf(parent_value) = node.borrow().content.clone() else {
        panic!("Cannot refine branch")
    };

    let mut has_neighbor = false;

    find_neighbors(&node, Edge::Bottom, &mut |_| {
        has_neighbor = true;
    });

    let value = if has_neighbor {
        parent_value
    } else {
        // Average value of the function
        let resolution = 2;
        sample_average(
            level + resolution,
            input_function.begin,
            input_function.end,
            |x| input_function.call(x),
        )// * 2_f32.powi(-(resolution as i32))
    };

    let branches = [(); 4].map(|_| Rc::new(RefCell::new(Node {
        parent: Some(node.clone()),
        level: level + 1,
        content: NodeContent::Leaf(value),
    })));

    node.borrow_mut().content = NodeContent::Branch(branches);
}

fn insert_function_rec(
    tree: NodeRef,
    max_residual_times_area: f32,
    max_level: usize,
    f: InputFunction,
) {
    // Can't go deeper than max level
    let level = tree.borrow().level;
    if level >= max_level {
        return;
    }

    let content = tree.borrow().content.clone();
    match content {
        NodeContent::Leaf(value) => {
            // Four steps
            let residual =
                sample_max(f.level + 5, f.begin, f.end, |x| (value - f.call(x)).abs());

            let area = 2_f32.powi(-(level as i32));
            if residual * area > max_residual_times_area {
            //if true {
                refine_cell(tree.clone(), f.clone());
                insert_function_rec(tree.clone(), max_residual_times_area, max_level, f);
            }
        }
        // Only insert on the bottom (least time) branches
        NodeContent::Branch(branches) => {
            insert_function_rec(
                branches[Quadrant::BotLeft as usize].clone(),
                max_residual_times_area,
                max_level,
                f.left(),
            );
            insert_function_rec(
                branches[Quadrant::BotRight as usize].clone(),
                max_residual_times_area,
                max_level,
                f.right(),
            );
        }
    }
}

fn sample_average(max_level: usize, begin: f32, end: f32, f: impl Fn(f32) -> f32) -> f32 {
    let step_size = 2_f32.powi(-(max_level as i32));
    let mut x = begin;
    let mut sum = 0.0;
    let mut steps = 0;
    while x < end {
        sum += f(x);
        x += step_size;
        steps += 1;
    }
    sum / (steps as f32).max(1.0)
}

fn sample_max(level: usize, begin: f32, end: f32, f: impl Fn(f32) -> f32) -> f32 {
    let step_size = 1.0 / level as f32;
    let mut x = begin;
    let mut max: f32 = 0.0;
    while x < end {
        max = max.max(f(x));
        x += step_size;
    }
    max
}

/// Calls f(min x, max x, value)
fn sample_grid_at_y(root: NodeRef, y: f32, f: &impl Fn(f32, f32, f32)) {
    sample_grid_at_y_rec(root, y, Rect::from_min_max(Pos2::ZERO, Pos2::new(1., 1.)), f);
}

fn sample_grid_at_y_rec(node: NodeRef, y: f32, rect: Rect, f: &impl Fn(f32, f32, f32)) {
    let content = node.borrow().content.clone();
    match content {
        NodeContent::Leaf(value) => {
            f(rect.min.x, rect.max.x, value)
        },
        NodeContent::Branch(branches) => {
            let (lefts, rights) = rect.split_left_right_at_fraction(0.5);
            let (top_left, bottom_left) = lefts.split_top_bottom_at_fraction(0.5);
            let (top_right, bottom_right) = rights.split_top_bottom_at_fraction(0.5);
            let rects = [top_left, top_right, bottom_left, bottom_right];

            for (branch, rect) in branches.into_iter().zip(rects) {
                if y <= rect.max.y && y > rect.min.y {
                    sample_grid_at_y_rec(branch, y, rect, f);
                }
            }
        }
    }
}

fn draw_func_at_y(tree: &NodeRef, ui: &mut Ui, disp_rect: Rect, y: f32, y_offset: f32, amplitude: f32) {
    sample_grid_at_y(tree.clone(), y, &|min_x, max_x, value| {
        ui.painter().line_segment([
            Pos2::new(disp_rect.min.lerp(disp_rect.max, min_x).x, value * amplitude + y_offset), 
            Pos2::new(disp_rect.min.lerp(disp_rect.max, max_x).x, value * amplitude + y_offset), 
        ], Stroke::new(1.0, Color32::GREEN));
    });
    ui.painter().line_segment([Pos2::new(disp_rect.min.x, y_offset), Pos2::new(disp_rect.max.x, y_offset)], Stroke::new(1.0, Color32::LIGHT_GRAY));
}

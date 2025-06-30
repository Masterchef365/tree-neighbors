use std::{cell::RefCell, rc::Rc};

use eframe::egui::{CentralPanel, Color32, Frame, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use rand::Rng;

fn main() {
    let mut tree = random_tree(5.0, None);

    eframe::run_simple_native("tree test", Default::default(), move |ctx, _frame| {
        CentralPanel::default().show(ctx, |ui| {
            Frame::canvas(ui.style()).show(ui, |ui| {
                draw_tree(ui, tree.clone());
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
}

#[derive(Clone)]
enum NodeContent {
    Branch([NodeRef; 4]),
    Leaf(f32),
}

fn random_tree(p: f64, parent: Option<NodeRef>) -> NodeRef {
    let new_node = Rc::new(RefCell::new(Node {
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

fn draw_tree(ui: &mut Ui, root: NodeRef) {
    let (rectangle, resp) = ui.allocate_exact_size(Vec2::splat(500.0), Sense::click_and_drag());
    zero_leaves(&root);

    // Show hover and neighbors
    if let Some(interact) = resp.hover_pos() {
        if let Some(found) = find_node_recursive(interact, root.clone(), rectangle) {
            if let NodeContent::Leaf(value) = &mut found.borrow_mut().content {
                *value = 0.75;
            }

            for edge in [Edge::Top, Edge::Bottom, Edge::Left, Edge::Right] {
                //eprintln!("BEGIN NEIGHBOR {edge:?}");
                neighbor_func(&found, edge, vec![], &mut |node| {
                    //eprintln!("MUT");
                    //debug_borrow!(node);
                    if let NodeContent::Leaf(value) = &mut node.borrow_mut().content {
                        *value = 0.20;
                    }
                });
                //eprintln!("END NEIGHBOR");
            }
        }
    }

    draw_tree_recursive(ui, &root, rectangle);
}

fn draw_tree_recursive(ui: &mut Ui, node: &NodeRef, rect: Rect) {
    match &node.borrow().content {
        NodeContent::Leaf(value) => {
            let fill = Color32::from_gray((255.0 * value) as u8);
            let stroke = Stroke::new(1., Color32::WHITE);
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

fn neighbor_func(
    node: &NodeRef,
    edge: Edge,
    mut up_tracking: Vec<Quadrant>,
    f: &mut impl FnMut(NodeRef),
) {
    //debug_borrow!(node);
    let Some(parent) = node.borrow().parent.clone() else {
        // Root has no neighbors
        return;
    };
    //debug_borrow!(parent);
    let NodeContent::Branch(branches) = &parent.borrow().content.clone() else {
        panic!("Invalid tree; parent node is a leaf!");
    };

    let quad = branches
        .iter()
        .position(|branch| Rc::ptr_eq(&node, branch))
        .map(Quadrant::from)
        .expect("Parent did not contain child");


    let branches = branches.clone();

    if let Some(adj) = adjacent_quadrant(quad, edge) {
        down_func(&branches[adj as usize], edge, up_tracking, f);
    } else {
        up_tracking.push(quad);
        neighbor_func(&parent, edge, up_tracking, f);
    }
}

fn down_func(
    node: &NodeRef,
    edge: Edge,
    mut up_tracking: Vec<Quadrant>,
    f: &mut impl FnMut(NodeRef),
) {
    //debug_borrow!(node);
    let content = node.borrow().content.clone();
    match content {
        NodeContent::Leaf(_) => f(node.clone()),
        NodeContent::Branch(branches) => {
            if let Some(quad) = up_tracking.pop() {
                let mirrored = quad.mirror(edge);
                down_func(&branches[mirrored as usize], edge, up_tracking, f);
            } else {
                for quad in edge.neighbor_quadrants() {
                    down_func(&branches[quad as usize], edge, up_tracking.clone(), f);
                }
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

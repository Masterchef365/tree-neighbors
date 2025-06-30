use std::{cell::RefCell, rc::Rc};

use eframe::egui::{CentralPanel, Color32, Frame, Pos2, Rect, Sense, Stroke, Ui, Vec2};
use rand::Rng;

fn main() {
    let mut tree = random_tree(130.0, None);

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

struct Node {
    parent: Option<NodeRef>,
    content: NodeContent,
}

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
        let content = NodeContent::Branch(
            [(); 4].map(|_| random_tree(p / 4.0, Some(new_node.clone()))),
        );
        new_node.borrow_mut().content = content;
    }

    new_node
}

fn draw_tree(ui: &mut Ui, root: NodeRef) {
    let (rectangle, resp) = ui.allocate_exact_size(Vec2::splat(500.0), Sense::click_and_drag());
    zero_leaves(&root);

    if let Some(interact) = resp.hover_pos() {
        if let Some(found) = find_node_recursive(interact, root.clone(), rectangle) {
            if let NodeContent::Leaf(value) = &mut found.borrow_mut().content {
                *value = 0.75;
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
        NodeContent::Leaf(_) => {
            rect.contains(pos).then(|| node.clone())
        }
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

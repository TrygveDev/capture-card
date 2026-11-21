//! The FPS counter over the video's top-left corner: frames on screen per second and the
//! game's own rate, or that with more numbers and frame-time graphs.

use gpui::{
    Bounds, Div, FontWeight, ParentElement, Pixels, SharedString, Styled, Window, canvas, div, fill, point,
    prelude::FluentBuilder, px, rgb, rgba, size,
};

use crate::meter::{COLUMNS, Graph, Numbers};
use crate::settings::Overlay;

/// Frame times on screen, and of the game's pictures.
const SCREEN_COLOR: u32 = 0x4ade80;
const GAME_COLOR: u32 = 0xfbbf24;
const GRAPH_HEIGHT: f32 = 40.0;

fn label(text: impl Into<SharedString>) -> Div {
    div().text_color(rgba(0xffffff8c)).child(text.into())
}

fn value(text: impl Into<SharedString>) -> Div {
    div().text_color(rgb(0xffffff)).font_weight(FontWeight::MEDIUM).child(text.into())
}

/// "name value", for a row of them.
fn stat(name: &'static str, v: String) -> Div {
    div().flex().gap(px(3.)).child(label(name)).child(value(v))
}

fn row() -> Div {
    div().flex().gap(px(10.)).text_size(px(12.)).whitespace_nowrap()
}

pub fn panel(mode: Overlay, n: &Numbers, graphs: Option<&(Graph, Option<Graph>)>) -> Div {
    let full = mode == Overlay::Graph;
    let headline = div()
        .flex()
        .items_baseline()
        .gap(px(4.))
        .whitespace_nowrap()
        .child(
            div()
                .text_size(px(28.))
                .line_height(px(30.))
                .font_weight(FontWeight::BOLD)
                .text_color(rgb(0xffffff))
                .child(format!("{:.0}", n.fps)),
        )
        .child(label("FPS").text_size(px(12.)).font_weight(FontWeight::BOLD))
        .when(full, |d| {
            d.child(div().flex_1().min_w(px(16.)))
                .child(value(format!("{:.1}", n.frame_ms)).text_size(px(12.)))
                .child(label("ms").text_size(px(12.)))
        });

    let mut rates = row();
    if let Some(game) = n.game {
        rates = rates.child(stat("game", format!("{game:.0}")));
    }
    if full {
        rates = rates.child(stat("card", format!("{:.0}", n.card)));
        if n.generated > 0.0 {
            rates = rates.child(stat("generated", format!("{:.0}%", n.generated * 100.0)));
        }
    }

    let mut timing = row();
    if let Some(low) = n.low {
        timing = timing.child(stat("1% low", format!("{low:.0}")));
    }
    if let Some(d) = n.delay_ms {
        timing = timing.child(stat("delay", format!("{d:.0} ms")));
    }
    // Only while frames are being generated; otherwise it's a guess from before.
    if let Some(g) = n.gpu_ms.filter(|_| n.generated > 0.0) {
        timing = timing.child(stat("GPU", format!("{g:.1} ms")));
    }

    div()
        .flex()
        .flex_col()
        .gap(px(2.))
        .px(px(10.))
        .py(px(7.))
        .rounded(px(8.))
        .bg(rgba(0x000000a6))
        .border_1()
        .border_color(rgba(0xffffff1a))
        .child(headline)
        .when(n.game.is_some() || full, |d| d.child(rates))
        .when(full, |d| d.child(timing))
        .when_some(graphs.filter(|_| full), |d, (screen, game)| {
            d.child(graph("ON SCREEN", SCREEN_COLOR, screen))
                .when_some(game.as_ref(), |d, g| d.child(graph("GAME", GAME_COLOR, g)))
        })
}

/// A labelled frame-time graph: time runs left to right over the last few seconds, height is
/// frame time up to the scale on the right.
fn graph(name: &'static str, color: u32, g: &Graph) -> Div {
    let g = g.clone();
    div()
        .flex()
        .flex_col()
        .gap(px(2.))
        .mt(px(5.))
        .child(
            div()
                .flex()
                .justify_between()
                .text_size(px(10.))
                .font_weight(FontWeight::BOLD)
                .child(div().text_color(rgb(color)).child(name))
                .child(label(format!("{:.0} ms", g.top))),
        )
        .child(
            canvas(|_, _, _| {}, move |b, (), window, _| paint_graph(b, &g, color, window))
                .w(px(COLUMNS as f32))
                .h(px(GRAPH_HEIGHT)),
        )
}

fn paint_graph(b: Bounds<Pixels>, g: &Graph, color: u32, window: &mut Window) {
    window.paint_quad(fill(b, rgba(0xffffff10)));
    let h = f32::from(b.size.height);
    let w = f32::from(b.size.width) / COLUMNS as f32;
    // Half the scale: a graph's usual frame time sits at or under this line.
    let half = Bounds::new(point(b.origin.x, b.origin.y + px(h / 2.0)), size(b.size.width, px(1.)));
    window.paint_quad(fill(half, rgba(0xffffff1f)));
    let y = |ms: f32| h * (1.0 - (ms / g.top).clamp(0.0, 1.0));
    for (i, c) in g.columns.iter().enumerate() {
        let Some((lo, hi)) = *c else { continue };
        // Steady frame times make a thin line; uneven ones a band from shortest to longest.
        let top = y(hi).min(h - 1.5);
        let bottom = y(lo).max(top + 1.5);
        let bar = Bounds::new(
            point(b.origin.x + px(i as f32 * w), b.origin.y + px(top)),
            size(px(w), px(bottom - top)),
        );
        window.paint_quad(fill(bar, rgb(color)));
    }
}

pub mod board;

use std::path::Path;

use board::{Board, Chunk};
use raylib::prelude::*;

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(1200,800)
        .title("Game of Life")
        .resizable()
        .build();

    let args = std::env::args().collect::<Vec<String>>();
    let static_board_file = if args.len() > 1 {
        &args[1]
    } else {
        "./board.gol"
    };

    /*
    Two boards stored: On-file static board (edit mode)
    Running board: Simulated from the static board (run mode)
    
    Static board is passed in from a binary file in command line arguments

    If a non-existent file was passed, a blank static board is generated
    and saved there on startup.

    If no file argument is passed, a blank static board is created
    at "./board.gol"
    */

    let mut static_board = Board::new();
    let mut running_board = Board::new();

    if Path::new(static_board_file).exists() {
        static_board.load(static_board_file);
    }

    // Mode is false when editing, true when running the simulation
    let mut mode = false;
    let cell_size = 4;
    let padding = 1;

    let mut click_dragging = false;
    let mut dragged_coords = vec![];

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);
        let (cx, cy) = (d.get_screen_width() / 2 - cell_size / 2, d.get_screen_height() / 2 - cell_size / 2);

        d.clear_background(Color::BLACK);

        let text = if !mode {
            "Edit Mode"
        } else {
            "Run Mode"
        };
        d.draw_text(text, 0, 0, 20, Color::GREEN);
        d.draw_text(&format!("Chunks: {}", if mode { running_board.chunk_count() } else { static_board.chunk_count() }), 0, 20, 20, Color::GREEN);

        if d.is_key_pressed(KeyboardKey::KEY_SPACE) {
            if !mode {
                running_board = static_board.clone();
            }

            mode = !mode;
        }

        if mode {
            if d.is_key_pressed(KeyboardKey::KEY_DOWN) {
                running_board.step_simulation();
            }

            if d.is_key_down(KeyboardKey::KEY_RIGHT) {
                running_board.step_simulation();
            }
        } else {
            if d.is_key_down(KeyboardKey::KEY_LEFT_CONTROL) && d.is_key_pressed(KeyboardKey::KEY_S) {
                static_board.save(static_board_file);
                println!("Saved board");
            }

            if click_dragging && d.is_mouse_button_up(MouseButton::MOUSE_LEFT_BUTTON) {
                click_dragging = false;
                dragged_coords.clear();
            }

            if d.is_mouse_button_down(MouseButton::MOUSE_LEFT_BUTTON) {
                if !click_dragging {
                    click_dragging = true;
                }

                let (bx, by) = get_board_coords(d.get_mouse_x(), d.get_mouse_y(), cx, cy, cell_size, d.get_screen_height());

                if !dragged_coords.contains(&(bx, by)) {
                    static_board.flip_cell(bx, by);
                    println!("mouse down: {} - {} - {}", bx, by, dragged_coords.len());
                    dragged_coords.push((bx, by));
                }
            }
        }

        
        let chunks = if mode { &running_board.chunks } else { &static_board.chunks };
        for (key, val) in chunks {
            for i in 0..8 {
                for j in 0..8 {
                    if !val.get_cell(i, j) {
                        continue;
                    }
                    let (gx, gy) = running_board.get_global_coords(key.0, key.1, i, j);

                    d.draw_rectangle(cx + gx * cell_size + padding, d.get_screen_height() - (cy + gy * cell_size + padding), cell_size - padding, cell_size - padding, Color::WHITE);
                }
            }
        }
    }
}

fn get_board_coords(mx: i32, my: i32, cx: i32, cy: i32, cell_size: i32, screen_height: i32) -> (i32, i32) {
    let x = mx - cx;
    let y = screen_height - cy - my;

    let mut bx = x / cell_size;
    let mut by = y / cell_size;

    if mx - cx < 0 {
        bx -= 1;
    }
    if my - cy < cell_size {
        by += 1;
    }

    (bx, by)
}

fn test_coord_system() {
    let board = Board::new();
    for i in -1000..1000 {
        for j in -1000..1000 {
            let (cx, cy, ix, iy) = board.get_chunk_coords(i, j);
            let (rx, ry) = board.get_global_coords(cx, cy, ix, iy);
            if i != rx || j != ry {
                println!("Failed: {} - {}, {} - {}", i, j, rx, ry);
            }
        }
    }
}

pub mod board;

use std::{fs, path::Path};

use board::{Board, Chunk};
use raylib::prelude::*;
use cust::prelude::*;

static PTX: &str = include_str!("gol.ptx");

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(1200,800)
        .title("Game of Life")
        .resizable()
        .build();

    let _ctx = cust::quick_init().unwrap();
    let gol_module = Module::from_ptx(PTX, &[]).unwrap();
    let stream = Stream::new(StreamFlags::NON_BLOCKING, None).unwrap();

    let cell_true_func = gol_module.get_function("check_true_cell").unwrap();
    let cell_false_func = gol_module.get_function("check_false_cell").unwrap();

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
    let mut auto_step = false;
    let mut cell_size = 40;
    let padding = 1;
    let mut simulation_step = 0;

    let mut click_dragging = false;
    let mut dragged_coords = vec![];

    let mut screen_offset_x = 0.0;
    let mut screen_offset_y = 0.0;
    let scroll_speed = 0.125;

    let mut file_string = String::new();

    if args.len() > 2 && args[2] == "-z" {
        let mut gpu = true;
        if args.len() > 3 && args[3] == "-c" {
            gpu = false;
        }
        running_board = static_board.clone();
        while !rl.window_should_close() {
            let mut d = rl.begin_drawing(&thread);
            d.clear_background(Color::BLACK);

            let fps = d.get_fps();
            d.draw_text(&format!("FPS: {}", fps), 0, 20, 20, Color::GREEN);
            d.draw_text(&format!("Simulation Step: {}", simulation_step), 0, 40, 20, Color::GREEN);
            let (t1, t2, t3, t4) = if gpu { running_board.step_simulation(&cell_true_func, &cell_false_func, &stream) } else { running_board.step_simulation_cpu() };
            simulation_step += 1;

            let total = t4 - t1;
            let cpu_pre = t2 - t1;
            let gpu = t3 - t2;
            let cpu_post = t4 - t3;
            
            let st = format!("#{}-Fps:{}: Total: {} - CPU Pre: {} - GPU: {} - CPU Post: {}", simulation_step, fps, total.as_micros(), cpu_pre.as_micros(), gpu.as_micros(), cpu_post.as_micros());
            //file_string = file_string + &format!("\n{}", st);
            println!("{}", st);
        }

        /*let fname = Path::new(&static_board_file).file_name().unwrap();
        fs::write(format!("./output_{}_{}.txt", if gpu { "gpu" } else { "cpu" }, fname.to_str().unwrap()), file_string).unwrap();*/
    } else {

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
        d.draw_text(&format!("FPS: {}", d.get_fps()), 0, 40, 20, Color::GREEN);
        d.draw_text(&format!("Simulation Step: {}", simulation_step), 0, 60, 20, Color::GREEN);

        if d.is_key_pressed(KeyboardKey::KEY_SPACE) {
            if !mode {
                running_board = static_board.clone();
            } else {
                simulation_step = 0;
                auto_step = false;
            }

            mode = !mode;
        }

        if d.is_key_down(KeyboardKey::KEY_W) {
            screen_offset_y += scroll_speed;
        }

        if d.is_key_down(KeyboardKey::KEY_LEFT_CONTROL) {
            if d.is_key_pressed(KeyboardKey::KEY_S) {
                static_board.save(static_board_file);
                println!("Saved board");
            }
        } else {
            if d.is_key_down(KeyboardKey::KEY_S) {
                screen_offset_y -= scroll_speed;
            }
        }
        if d.is_key_down(KeyboardKey::KEY_A) {
            screen_offset_x += scroll_speed;
        }
        if d.is_key_down(KeyboardKey::KEY_D) {
            screen_offset_x -= scroll_speed;
        }

        let scrolled = d.get_mouse_wheel_move();
        let scale_factor = std::cmp::max(2, cell_size / 15) as f32;
        if scrolled < 0.0 && cell_size >= 6 {
            cell_size += (scrolled * scale_factor) as i32;
        }
        if scrolled > 0.0 && cell_size <= 90 {
            cell_size += (scrolled * scale_factor) as i32;
        }

        if mode {
            if d.is_key_pressed(KeyboardKey::KEY_PERIOD) {
                auto_step = !auto_step;
            }

            if d.is_key_pressed(KeyboardKey::KEY_DOWN) {
                running_board.step_simulation(&cell_true_func, &cell_false_func, &stream);
                simulation_step += 1;
            }

            if d.is_key_down(KeyboardKey::KEY_RIGHT) || auto_step {
                running_board.step_simulation(&cell_true_func, &cell_false_func, &stream);
                simulation_step += 1;
            }
        } else {
            if click_dragging && d.is_mouse_button_up(MouseButton::MOUSE_LEFT_BUTTON) {
                click_dragging = false;
                dragged_coords.clear();
            }

            if d.is_mouse_button_down(MouseButton::MOUSE_LEFT_BUTTON) {
                if !click_dragging {
                    click_dragging = true;
                }

                let (bx, by) = get_board_coords(d.get_mouse_x(), d.get_mouse_y(), cx, cy, cell_size, d.get_screen_height(), screen_offset_x, screen_offset_y);

                if !dragged_coords.contains(&(bx, by)) {
                    static_board.flip_cell(bx, by);
                    dragged_coords.push((bx, by));
                }
            }
        }
        
        let cells = if mode { &running_board.true_positions } else { &static_board.true_positions };
        for (x, y) in cells {
            d.draw_rectangle(cx + x * cell_size + padding + screen_offset_x as i32, d.get_screen_height() - (cy + y * cell_size + padding) + screen_offset_y as i32, cell_size - padding, cell_size - padding, Color::WHITE);
        }
    }
    }
}

fn get_board_coords(mx: i32, my: i32, cx: i32, cy: i32, cell_size: i32, screen_height: i32, offset_x: f32, offset_y: f32) -> (i32, i32) {
    let x = mx - cx - offset_x as i32;
    let y = screen_height - cy - my + offset_y as i32;

    let mut bx = x / cell_size;
    let mut by = y / cell_size;

    if mx - cx - (offset_x as i32) < 0 {
        bx -= 1;
    }
    if my - cy - (offset_y as i32) < cell_size {
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

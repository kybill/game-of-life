pub mod board;

use board::{Board, Chunk};
use raylib::prelude::*;

fn main() {
    let (mut rl, thread) = raylib::init()
        .size(1200,800)
        .title("Game of Life")
        .resizable()
        .build();

    rl.set_target_fps(60);

    /*
    Two boards stored: On-file static board (edit mode)
    Running board: Simulated from the static board (run mode)
    
    Static board is passed in from a binary file in command line arguments
    */

    let mut board = Board::new();
    /*board.get_chunk_coords(0, 0);
    board.get_chunk_coords(15, 15);
    board.get_chunk_coords(16, 16);
    board.get_chunk_coords(-1, -1);
    board.get_chunk_coords(-13, -8);
    board.get_chunk_coords(-1, -1);
    board.get_chunk_coords(-16, -16);*/

    board.set_cell(-1, -1, true);
    board.set_cell(0, 0, true);
    board.set_cell(1, 1, true);
    board.set_cell(2, 2, true);

    let chunk = board.get_chunk(0, 0).unwrap();
    println!("{:0b}", chunk.get_raw());
    println!("{} - {} - {}", chunk.get_cell(0, 0), chunk.get_cell(1, 1), chunk.get_cell(2, 2));

    let (array, conv_x, conv_y) = board.generate_array();
    for sub_array in array {
        println!("{:?}", sub_array);
    }

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
    }
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

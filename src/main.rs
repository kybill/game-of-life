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

    let board = Board::new();
    board.get_chunk_coords(0, 0);
    board.get_chunk_coords(15, 15);
    board.get_chunk_coords(16, 16);
    board.get_chunk_coords(-1, -1);
    board.get_chunk_coords(-13, -8);
    board.get_chunk_coords(-1, -1);
    board.get_chunk_coords(-16, -16);

    while !rl.window_should_close() {
        let mut d = rl.begin_drawing(&thread);

        d.clear_background(Color::WHITE);
    }
}
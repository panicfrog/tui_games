// main.rs
mod maze;

use crossterm::event::{self, Event, KeyCode};
use crossterm::terminal::{enable_raw_mode, disable_raw_mode};
use ratatui::{
    backend::CrosstermBackend,
    layout::{Constraint, Layout},
    style::{Color, Style},
    text::{Line, Span, Text},
    widgets::{Block, Borders, Paragraph},
    Terminal,
};
use std::{error::Error, io, time::Duration};

use maze::{Maze, Position, TileType};

struct Game {
    maze: Maze,
    player: Position,
    finished: bool,
}

impl Game {
    fn new(width: usize, height: usize) -> Self {
        let maze = Maze::new(width, height);
        Game {
            player: maze.start,
            maze,
            finished: false,
        }
    }

    fn try_move(&mut self, dx: isize, dy: isize) {
        if self.finished {
            return;
        }
        let nx = self.player.x as isize + dx;
        let ny = self.player.y as isize + dy;

        if nx >= 0
            && ny >= 0
            && nx < self.maze.width as isize
            && ny < self.maze.height as isize
        {
            let c = self.maze.grid[ny as usize][nx as usize];
            if c != TileType::Wall {
                self.player.x = nx as usize;
                self.player.y = ny as usize;
                if c == TileType::Exit {
                    self.finished = true;
                }
            }
        }
    }

    fn render(&self) -> Text {
        let mut lines = Vec::new();
        for (y, row) in self.maze.grid.iter().enumerate() {
            let mut spans = Vec::new();
            for (x, &ch) in row.iter().enumerate() {
                let is_player = self.player.x == x && self.player.y == y;
                let style = if is_player {
                    Style::default().fg(Color::Green).bg(Color::Black)
                } else {
                    match ch {
                        TileType::Wall => Style::default().bg(Color::White),
                        TileType::Exit => Style::default().fg(Color::Black).bg(Color::Yellow),
                        TileType::Path => Style::default().bg(Color::DarkGray),
                    }
                };

                let display_char = match (is_player, ch) {
                    (true, _) => '@',
                    (false, TileType::Exit) => 'E',
                    _ => ' ',
                };

                spans.push(Span::styled(display_char.to_string(), style));
            }
            lines.push(Line::from(spans));
        }
        Text::from(lines)
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    let backend = CrosstermBackend::new(&mut stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut game = Game::new(41, 21); // 可调节尺寸

    loop {
        if game.finished {
            terminal.draw(|f| {
                let para = Paragraph::new("🎉 You reached the goal!\nPress 'q' to quit.")
                    .block(Block::default().borders(Borders::ALL).title("Victory"))
                    .style(Style::default());
                f.render_widget(para, f.area());
            })?;
        } else {
            terminal.draw(|f| {
                let layout = Layout::default()
                    .constraints([Constraint::Min(0)])
                    .split(f.area());
                let para = Paragraph::new(game.render())
                    .block(Block::default().borders(Borders::ALL).title("Maze"))
                    .style(Style::default());
                f.render_widget(para, layout[0]);
            })?;
        }

        if event::poll(Duration::from_millis(100))? {
            if let Event::Key(key) = event::read()? {
                match key.code {
                    KeyCode::Up => game.try_move(0, -1),
                    KeyCode::Down => game.try_move(0, 1),
                    KeyCode::Left => game.try_move(-1, 0),
                    KeyCode::Right => game.try_move(1, 0),
                    KeyCode::Char('q') => break,
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    Ok(())
}

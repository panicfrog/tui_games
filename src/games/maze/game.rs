use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span, Text};

use super::map::{Action, MazeMap, MazeStatus, Position, TileType};
use crate::env::Env;

pub struct Game {
    pub maze: MazeMap,
    player: Position,
    pub finished: bool,
    level: usize,
    steps: usize,
    last_action: Option<Action>, // ✅ 新增：用于状态判断
    selected_button: VictoryButton,
}

#[derive(Clone, Copy)]
pub enum VictoryButton {
    Quit,
    Next,
}

#[derive(Debug, Clone, Copy)]
pub enum InputAction {
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    Quit,
    Confirm,
    ToggleButton,
}

impl Game {
    pub fn new(width: usize, height: usize) -> Self {
        let maze = MazeMap::new(width, height);
        Game {
            player: maze.start,
            maze,
            finished: false,
            level: 1,
            steps: 0,
            last_action: None,
            selected_button: VictoryButton::Next,
        }
    }

    pub fn handle_action(&mut self, action: InputAction) -> bool {
        use InputAction::*;
        if self.finished {
            match action {
                ToggleButton => {
                    self.selected_button = match self.selected_button {
                        VictoryButton::Quit => VictoryButton::Next,
                        VictoryButton::Next => VictoryButton::Quit,
                    };
                }
                Confirm => match self.selected_button {
                    VictoryButton::Quit => return true,
                    VictoryButton::Next => self.next_level(),
                },
                _ => {}
            }
            return true;
        } else {
            match action {
                MoveUp => self.try_move(Action::Up),
                MoveDown => self.try_move(Action::Down),
                MoveLeft => self.try_move(Action::Left),
                MoveRight => self.try_move(Action::Right),
                Quit => return true,
                _ => {}
            }
        }
        false
    }

    fn try_move(&mut self, action: Action) {
        let (dx, dy) = action.delta();
        let nx = self.player.x as isize + dx;
        let ny = self.player.y as isize + dy;

        if nx >= 0 && ny >= 0 && nx < self.maze.width as isize && ny < self.maze.height as isize {
            let tile = self.maze.grid[ny as usize][nx as usize];
            if tile != TileType::Wall {
                self.player.x = nx as usize;
                self.player.y = ny as usize;
                self.steps += 1;
                self.last_action = Some(action);
                if tile == TileType::Exit {
                    self.finished = true;
                }
            }
        }
    }

    fn next_level(&mut self) {
        self.level += 1;
        self.maze = MazeMap::new(self.maze.width + 2, self.maze.height + 2);
        self.player = self.maze.start;
        self.finished = false;
        self.steps = 0;
        self.last_action = None;
        self.selected_button = VictoryButton::Next;
    }

    pub fn render(&self) -> (Text, &'static str) {
        if self.finished {
            (self.render_victory_screen(), "Victory")
        } else {
            (self.render_maze(), "Maze")
        }
    }

    fn render_maze(&self) -> Text {
        let mut lines = Vec::new();
        for (y, row) in self.maze.grid.iter().enumerate() {
            let mut spans = Vec::new();
            for (x, &tile) in row.iter().enumerate() {
                let is_player = self.player.x == x && self.player.y == y;
                let style = if is_player {
                    Style::default().fg(Color::Green).bg(Color::Black)
                } else {
                    match tile {
                        TileType::Wall => Style::default().bg(Color::White),
                        TileType::Exit => Style::default().fg(Color::Black).bg(Color::Yellow),
                        TileType::Path => Style::default().bg(Color::DarkGray),
                    }
                };

                let display_char = match (is_player, tile) {
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

    fn render_victory_screen(&self) -> Text {
        let quit_style = if matches!(self.selected_button, VictoryButton::Quit) {
            Style::default().fg(Color::Black).bg(Color::Red)
        } else {
            Style::default().fg(Color::White).bg(Color::DarkGray)
        };
        let next_style = if matches!(self.selected_button, VictoryButton::Next) {
            Style::default().fg(Color::Black).bg(Color::Green)
        } else {
            Style::default().fg(Color::White).bg(Color::DarkGray)
        };

        Text::from(vec![
            Line::from("🎉 You reached the goal!"),
            Line::from(""),
            Line::from(vec![
                Span::styled(" 结束游戏 ", quit_style),
                Span::raw("    "),
                Span::styled(" 继续挑战 ", next_style),
            ]),
        ])
    }
}

impl Env for Game {
    type State = Position;
    type Action = Action;
    type Status = MazeStatus;

    fn reset(&mut self) -> Self::State {
        self.player = self.maze.start;
        self.steps = 0;
        self.last_action = None;
        self.finished = false;
        self.player
    }

    fn step(&mut self, action: Action) -> (Self::State, Self::Status) {
        self.try_move(action);
        let (_, status) =
            self.maze
                .allowed_actions_and_status(self.player, self.last_action, self.steps);
        (self.player, status)
    }

    fn current_state(&self) -> Self::State {
        self.player
    }

    fn legal_actions(
        &self,
        state_and_action: Option<(Self::State, Self::Action)>,
    ) -> Vec<Self::Action> {
        let sa = if let Some((state, action)) = state_and_action {
            (state, Some(action))
        } else {
            (self.player, self.last_action)
        };
        let (actions, _) = self.maze.allowed_actions_and_status(sa.0, sa.1, self.steps);

        actions
    }
    fn is_terminal(&self) -> bool {
        match self
            .maze
            .allowed_actions_and_status(self.player, self.last_action, self.steps)
        {
            (_, MazeStatus::Ongoing) => false,
            (_, MazeStatus::Win) => true,
            (_, MazeStatus::FailDeadend) => true,
            (_, MazeStatus::FailStepLimit) => true,
        }
    }
    fn is_win(&self) -> bool {
        match self
            .maze
            .allowed_actions_and_status(self.player, self.last_action, self.steps)
        {
            (_, MazeStatus::Win) => true,
            _ => false,
        }
    }
}

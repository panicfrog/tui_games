pub trait Env {
    type State;
    type Action;
    type Status;

    #[allow(dead_code)]
    fn reset(&mut self) -> Self::State;
    #[allow(dead_code)]
    fn step(&mut self, action: Self::Action) -> (Self::State, Self::Status);
    #[allow(dead_code)]
    fn current_state(&self) -> Self::State;
    #[allow(dead_code)]
    fn legal_actions(&self) -> Vec<Self::Action>;
    #[allow(dead_code)]
    fn is_terminal(&self) -> bool;
    #[allow(dead_code)]
    fn is_win(&self) -> bool;
}

package cs3500.connectn;

public class ConnectN {

  private int turn;
  private int board[][];
  private boolean gameStarted;
  private boolean isGameOver;
  private int goal;
  private int numPlayers;

  private ConnectN(int width, int height, int goal, int numPlayers) {
    this.board = new int[height][width];
    this.goal = goal;
    this.numPlayers = numPlayers;
    //... init the rest ...
  }

  public static ConnectNBuilder builder() {
    return new ConnectNBuilder();
  }

  public static class ConnectNBuilder {

    public ConnectNBuilder() {

    }

  }
}


//ConnectNBuilder builder = new ConnectNBuilder();
//model = builder.build(); // default Connect 4

//builder.setGoal(3);
//model = builder.build(); // Connect 3 game with default board and 2 players

//builder = new ConnectNBuilder();
//builder.setWidth(3);
//builder.setHeight(4);
//model = builder.build();

//or add on top of the above with
//builder.setWidth(3);
//builder.setHeight(3);
//builder.setGoal(3);
//model = builder.build(); // TicTacToe with gravity


//Actual builder use
// model = new ConnectNBuilder().setWidth(3).setGoal(4).build();





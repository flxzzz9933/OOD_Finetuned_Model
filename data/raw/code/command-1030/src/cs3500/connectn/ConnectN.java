package cs3500.connectn;

public class ConnectN {

  private int width, height;
  private int goal;
  private int numPlayers;

  public ConnectN(int width, int height, int goal, int numPlayers) {
    this.width = width;
    this.height = height;
    this.goal = goal;
    this.numPlayers = numPlayers;
  }

  public static ConnectNBuilder builder() {
    return new ConnectNBuilder();
  }

  public static class ConnectNBuilder {
    private int width, height, goal, numPlayers;

    public ConnectNBuilder() {
      this.width = 7;
      this.height = 8;
      this.goal = 4;
      this.numPlayers = 2;
    }

    public ConnectN build() {
      return new ConnectN(width, height, goal, numPlayers);
    }

    public ConnectNBuilder setWidth(int width) {
      this.width = width;
      return this;
    }

    public ConnectNBuilder setHeight(int height) {
      this.height = height;
      return this;
    }

    public ConnectNBuilder setGoal(int goal) {
      this.goal = goal;
      return this;
    }

    public ConnectNBuilder setNumPlayers(int numPlayers) {
      this.numPlayers = numPlayers;
      return this;
    }
  }
}


//ConnectNBuilder usage

//ConnectNBuilder builder = new ConnectNBuilder();
//ConnectN model = builder.build(); //full default

//game with goal length of  3
//builder.setGoal(3);
//builder.build();

// new ConnectNBuilder().setGoal(3).setNumPlayer(5).build();



package view;

public interface IGUIView {

  String getInput();

  void displayText(String text);

  void clearInput();
  
  void setViewActions(ViewActions actions);

  void makeVisible();
}

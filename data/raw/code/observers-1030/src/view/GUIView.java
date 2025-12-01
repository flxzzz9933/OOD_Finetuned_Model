package view;

public interface GUIView {

  String getInput();

  //Could also take in no args and have the view delegate to the model
  void displayText(String string);

  void clearInput();

  void setViewActions(ViewActions actions);
}

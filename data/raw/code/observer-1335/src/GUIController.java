import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

import model.IModel;
import view.IGUIView;
import view.ViewActions;

public class GUIController implements ViewActions {

  private final IModel model;
  private final IGUIView view;

  public GUIController(IModel model, IGUIView view) {
    this.model = model;
    this.view = view;
  }


  @Override
  public void displayText(String text) {
    model.setString(text);
    view.displayText(model.getString());
    view.clearInput();
  }

  @Override
  public void exitProgram() {
    System.exit(0);
  }

  public void runProgram() {
    view.setViewActions(this);
    view.makeVisible();
  }
}

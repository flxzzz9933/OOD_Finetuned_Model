import view.GUIView;
import view.ViewActions;


public class GUIController implements ViewActions {

  private final IModel model;
  private final GUIView view;

  public GUIController(IModel model, GUIView view) {
    this.model = model;
    this.view = view;
    view.setViewActions(this);
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
}

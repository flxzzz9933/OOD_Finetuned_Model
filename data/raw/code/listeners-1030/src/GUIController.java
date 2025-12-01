import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

/**
 * As we saw, this is bad design. This is tied to ActionListener, meaning it's tied
 * to a view with buttons in Swing. If we change that view, this controller must change.
 * But how do we solve this issue? (Observer Pattern)
 */
public class GUIController implements ActionListener {

  private final IModel model;
  private final GUIView view;

  public GUIController(IModel model, GUIView view) {
    this.model = model;
    this.view = view;
  }

  //REACTS to an ActionEvent, like a button click
  @Override
  public void actionPerformed(ActionEvent e) {
    switch (e.getActionCommand()) {
      case "echo":
        String input = view.getInput();
        model.setString(input);
        view.displayText(model.getString());
        view.clearInput();
        break;
      case "exit":
        System.exit(0);
      default:
        new RuntimeException("Unknown action command: " + e.getActionCommand());
    }
  }
}

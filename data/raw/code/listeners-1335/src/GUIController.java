import java.awt.event.ActionEvent;
import java.awt.event.ActionListener;

public class GUIController implements ActionListener {

  private final IModel model;
  private final IGUIView view;

  public GUIController(IModel model, IGUIView view) {
    this.model = model;
    this.view = view;
  }


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
    }
  }
}

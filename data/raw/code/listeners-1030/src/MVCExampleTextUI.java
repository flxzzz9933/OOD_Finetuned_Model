
public class MVCExampleTextUI {
    public static void main(String []args) {
        IModel model = new Model();
        EchoFrame view = new EchoFrame(model); //BAD: Not the interface type!! How do we fix?
        GUIController controller = new GUIController(model, view);
        view.setVisible(true);

        /*
         * // OLD WAY!
        IModel model = new Model();
        IView view = new TextView(System.out);
        IController controller = new TextController(model,System.in,view);
        controller.runProgram();
         */
    }
}

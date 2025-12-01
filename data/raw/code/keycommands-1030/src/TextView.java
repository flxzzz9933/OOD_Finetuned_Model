import java.io.PrintStream;

public class TextView implements IView {
    private PrintStream out;

    public TextView(PrintStream out) {
        this.out = out;
    }

    public void showString(String s) {
        out.println("String: "+s);
    }

    public void showOptions() {
        //print the UI
        out.println("Menu: ");
        out.println("E: Enter a string");
        out.println("Q: Quit the program");
        out.print("Enter your choice: ");
    }

    public void showStringEntry() {
        out.print("\nEnter the string to be echoed: ");
    }

    public void showOptionError() {
        out.print("\nInvalid option. Please try again.");
    }

}

package game;
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
// import java.awt.Color;

public class Finish extends JPanel implements  ActionListener{
  private javax.swing.Timer timer;
  // private JPanel Panel1 = new JPanel();
  private int clicked=0; //0:押されてない, 1:continue, 2: StageSelect

  public Finish(int state){
    JButton Continue = new JButton("Continue");
    JButton StageSelect = new JButton("Title");
    JLabel str_state = new JLabel();
    str_state.setHorizontalAlignment(JLabel.CENTER);
    JPanel Panel1 = new JPanel(new GridLayout(1,2));
    if(state == -1) str_state.setText("GAMEOVER");
    if(state == 1) str_state.setText("CLEAR");
    str_state.setFont(new Font("ＭＳ ゴシック", Font.BOLD, 45));
    str_state.setForeground(Color.WHITE);
    setBackground(Color.BLACK);
    Panel1.setBackground(Color.BLACK);

    Continue.addActionListener(this);
    Continue.setFont(new Font("Arial", Font.PLAIN, 25));
    Continue.setActionCommand("Continue");

    StageSelect.addActionListener(this);
    StageSelect.setFont(new Font("Arial", Font.PLAIN, 25));
    StageSelect.setActionCommand("Title");

    Panel1.add(Continue);
    Panel1.add(StageSelect);
    setPreferredSize(new Dimension(301,601));
    setFocusable(true);
    // Panel1.setLayout(new GridLayout(2, 3));
    // // Panel1.setBackground(Color.BLACK);
    // // Panel1.add(str_state);
    // // Panel1.add(BackTitle);
    setLayout(new BorderLayout(100,100));
    add(str_state,BorderLayout.CENTER);
    add(Panel1,BorderLayout.PAGE_END);
    // add(Panel1);
  }

  public int getClicked(){
    return this.clicked;
  }

  public void actionPerformed(ActionEvent e){
    if(e.getActionCommand() != 	null){
      String cmd = e.getActionCommand();
      if(cmd.equals("Continue")){
        clicked = 1;
      }
      if(cmd.equals("Title")){
        clicked = 2;
      }
    }
  }
}

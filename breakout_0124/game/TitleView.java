package game;
// import game.block.*;
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

public class TitleView extends JPanel implements  ActionListener{
  private javax.swing.Timer timer;
  private int clicked=0;//0:押してない、1:play, 2:make
  public TitleView(){
    JPanel panel = new JPanel(new GridLayout(1,2));
    JButton playButton = new JButton("Play");
    JButton makeButton = new JButton("Make");
    JLabel title = new JLabel(new ImageIcon("./game/title-1.png"));
    // JLabel str_Title = new JLabel("BreakOut");
    playButton.addActionListener(this);
    playButton.setActionCommand("play");
    makeButton.addActionListener(this);
    makeButton.setActionCommand("make");
    playButton.setFont(new Font("Arial", Font.PLAIN, 25));
    makeButton.setFont(new Font("Arial", Font.PLAIN, 25));
    // str_Title.setFont(new Font("ＭＳ ゴシック", Font.BOLD, 50));
    panel.add(playButton); panel.add(makeButton);
    setPreferredSize(new Dimension(301,601));
    setFocusable(true);
    // clicked = false;
    setBackground(Color.BLACK);
    // str_Title.setHorizontalAlignment(JLabel.CENTER);
    // str_Title.setForeground(Color.WHITE);
    panel.setBackground(Color.BLACK);
    panel.setPreferredSize(new Dimension(300, 50));
    setLayout(new BorderLayout());
    add(title,BorderLayout.CENTER);
    add(panel,BorderLayout.PAGE_END);
  }

  public int getClicked(){
    return this.clicked;
  }

  public void actionPerformed(ActionEvent e){
    if(e.getActionCommand() != 	null){
      String cmd = e.getActionCommand();
      if(cmd.equals("play")){
        clicked = 1;
      }
      if(cmd.equals("make")){
        clicked = 2;
      }
    }
  }
}

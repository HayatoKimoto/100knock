package game;
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;
import javax.swing.border.*;

public class StageSelectView extends JPanel implements  ActionListener{
  private JPanel cardPanel;
  private int stage;// 0:未選択、1,2,3,..ステージ番号
  public StageSelectView(){
    cardPanel = new JPanel();
    setBackground(Color.BLACK);
    JButton Stage1 = new JButton("<html><img src='file:stage1.png' width=90 height=80></html>");
    JButton Stage2 = new JButton("<html><img src='file:stage2.png' width=90 height=80></html>");
    JButton Stage3 = new JButton("<html><img src='file:stage3.png' width=90 height=80></html>");
    JButton Stage4 = new JButton("<html><img src='file:stage4.png' width=90 height=80></html>");
    JButton Stage5 = new JButton("<html><img src='file:stage5.png' width=90 height=80></html>");
    JButton Stage6 = new JButton("<html><img src='file:user.png' width=90 height=80></html>");
    JLabel intro = new JLabel("<html><img src='file:introduce.png' width=300 height=200></html>");


    Stage1.setBackground(Color.BLACK);
    Stage2.setBackground(Color.BLACK);
    Stage3.setBackground(Color.BLACK);
    Stage4.setBackground(Color.BLACK);
    Stage5.setBackground(Color.BLACK);
    Stage6.setBackground(Color.BLACK);
    intro.setBackground(Color.BLACK);
    cardPanel.setBackground(Color.BLACK);


    Stage1.setBorderPainted(false);
    Stage2.setBorderPainted(false);
    Stage3.setBorderPainted(false);
    Stage4.setBorderPainted(false);
    Stage5.setBorderPainted(false);
    Stage6.setBorderPainted(false);

    Stage1.addActionListener(this);
    Stage2.addActionListener(this);
    Stage3.addActionListener(this);
    Stage4.addActionListener(this);
    Stage5.addActionListener(this);
    Stage6.addActionListener(this);

    Stage1.setActionCommand("1");
    Stage2.setActionCommand("2");
    Stage3.setActionCommand("3");
    Stage4.setActionCommand("4");
    Stage5.setActionCommand("5");
    Stage6.setActionCommand("6");

    setPreferredSize(new Dimension(301,601));
    setFocusable(true);
    stage = 0;

    cardPanel.add(Stage1);
    cardPanel.add(Stage2);
    cardPanel.add(Stage3);
    cardPanel.add(Stage4);
    cardPanel.add(Stage5);
    cardPanel.add(Stage6);
    setLayout(new GridLayout(2,1));
    add(cardPanel);
    add(intro);
  }

  public int getStage(){
    return this.stage;
  }
  public void setStage(int a){
    stage=a;
  }

  public void actionPerformed(ActionEvent e){
    if(e.getActionCommand() != 	null){
      String cmd = e.getActionCommand();
      if(cmd.equals("1")){
        setStage(1);
      }
      if(cmd.equals("2")){
        setStage(2);
      }
      if(cmd.equals("3")){
        setStage(3);
      }
      if(cmd.equals("4")){
        setStage(4);
      }
      if(cmd.equals("5")){
        setStage(5);
      }
      if(cmd.equals("6")){
        setStage(6);
      }
    }
  }
}

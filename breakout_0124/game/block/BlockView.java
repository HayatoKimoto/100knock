package game.block;
import java.awt.*;
import javax.swing.*;
import java.awt.event.*;

public class BlockView extends JPanel implements ActionListener{
  private javax.swing.Timer timer;
  private BlockModel bm;
  private int i;
  private int judge;
  public BlockView(){
    this.bm = new BlockModel();
    judge = 0;
  }

  public void paintComponent(Graphics g) {
    super.paintComponent(g);
    bm.draw(g);
  }

  public void actionPerformed(ActionEvent e){
    i = bm.check();
    if(i>=0){ bm.delete(i); }

    if(bm.clear_check()){
      game_clear();
      timer.stop();
    }
    repaint();
  }

  public void start(int a){
    timer = new javax.swing.Timer(100,this);
    timer.start();
    bm.make_block(a);
  }

  public void game_clear(){ judge=1; }

  public void game_over(){ judge=-1; }

  public void setJudge(int i){ this.judge=i;}
  public int getJudge(){ return judge; }
}

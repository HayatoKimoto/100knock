package game;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import game.*;
import game.block.*;
import java.util.*;
import game.item.*;

public class GameView extends JPanel implements MouseListener,MouseMotionListener,ActionListener{
   private javax.swing.Timer timer;
   private CourtModel tm;
   private BlockModel bm;
   private boolean ballMoving;
   final static int court_size_x=300;
   final static int court_size_y=600;
   private int game_state = 0; //-1:gameover,0:playing,1:clear
   private int click=0;
   private int residue=3; //残基
   private boolean click_state=false;
   private JLabel res_label;
   private JLabel item_label;
   private JLabel speed_label;
   private int time_count;
   public GameView(CourtModel tm){
     this.tm=tm;
     this.bm = new BlockModel();
     ballMoving=false;
     setBackground(Color.BLACK);
     setPreferredSize(new Dimension(tm.court_size_x,tm.court_size_y));
     setFocusable(true);
     addMouseMotionListener(this);
     addMouseListener(this);
     timer = new javax.swing.Timer(10,this); // 10ミリ秒ごとにボールが移動
     this.setLayout(null);
     res_label = new JLabel("×"+residue);
     item_label = new JLabel();
     speed_label = new JLabel();
     res_label.setForeground(Color.WHITE);
     item_label.setForeground(Color.WHITE);
     speed_label.setForeground(Color.WHITE);
     res_label.setBounds(230, 490, 200, 200);
     item_label.setBounds(120,490,200,200);
     speed_label.setBounds(10,490,200,200);
     this.add(res_label);
     this.add(item_label);
     this.add(speed_label);
   }

   public void setBlockModel(int a){
     bm.make_block(a);
     timer.start();
   }

   //ドラッグには反応しない
   public void mouseDragged(MouseEvent e){
   }
   // アプレットの上でマウスが動くのに合わせてコントローラを移動
   public void mouseMoved(MouseEvent e){
     ///マウスのx座標を取得する
      Point point = e.getPoint();
    if(click_state){
      tm.myself.move(point.x);
      // System.out.println("mouse.x : " + point.x);
     }else{
      tm.setLine(point.x, point.y);
     }
     repaint();
   }

   protected void paintComponent(Graphics g){
     super.paintComponent(g);
     tm.draw(g);
     bm.draw(g);
     if(!click_state){
       tm.Linedraw(g);
     }
   }

   public int getState(){
     return this.game_state;
   }

   public void actionPerformed(ActionEvent e){
    if(!tm.isInCourt()){
      tm.myself.settype(0);
      click=0;
      click_state=false;
      tm.initialize();
      setColor(0);
      if(residue==0){
        game_state = -1;
        timer.stop();
      }
    }else{
      //ボールが当たったブロックの配列の添字を返す
      int i = bm.check(tm.getx0(),tm.gety0());
      if(i >= 0)bm.delete(i);
    }

    if(bm.clear_check()){ //クリアチェック
      game_state = 1;
      timer.stop();
    }

    for(Item i: tm.item_list){
      i.moveDown();
    }

    for(Ballet bal: tm.ballet_list){
      bal.moveUp();
    }

    int type = tm.getItem();
    bm = tm.balletHit(bm);
    // tm.bm = bm;
    setColor(type);
if(type!=-1) System.out.println(type);
      if(type==0) tm.setTimer_count(0);
      if(tm.getTimer_count()<10)
        item_label.setText("00:0"+tm.getTimer_count());
      else
        item_label.setText("00:"+tm.getTimer_count());
     
    if(click_state){
      tm.moveBall();
      time_count++;
    }

    if(time_count==100){
      tm.setBallSpeedUp();
      time_count=0;
    }

    speed_label.setText("speed="+tm.getSpeed());
    repaint();
   }

  public void setColor(int type){
    if(type==0) item_label.setForeground(Color.WHITE);
    else if(type==1) item_label.setForeground(Color.GREEN);
    else if(type==2) item_label.setForeground(Color.RED);
  }


  public void mousePressed(MouseEvent e) {
    if(!click_state) residue -=1;
    else if(tm.myself.getType()==2)
      tm.ballet_list.add(new Ballet(tm.myself.getX()+tm.myself.getWidth()/2-1,tm.myself.getY()-tm.myself.getHeight()/2-6));
    res_label.setText("×"+residue);
    click += 1;
    if(click==1) click_state=true;
  }
  public void mouseClicked(MouseEvent e) { }
  public void mouseReleased(MouseEvent e){ }
  public void mouseEntered(MouseEvent e) { }
  public void mouseExited(MouseEvent e)  { }
}

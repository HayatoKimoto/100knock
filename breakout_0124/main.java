import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import game.*;
import game.block.*;
import java.util.*;
import java.io.File;
import java.net.MalformedURLException;
import javafx.scene.media.AudioClip;

// View
//   Controllerはカーソルの左右だけなので，独立させないでViewの内部に実装

class main extends JFrame {
   public static void main(String[] args) throws Exception{
    String str;
    CourtModel tm;
    BlockModel bm;
    AudioClip bgm=new AudioClip(new File("bgm_maoudamashii_8bit14.mp3").toURI().toString());
    bgm.setVolume(0.3);
    int state=0;
    int stage_num=1;
    int  nextView = 0; // 0:Title, 1:StageSelect, 2:GameView, 3:finish
    boolean click_state;
    int click_state_int;
    JFrame frame  = new JFrame("Break Out");
    // System.out.println("Start");
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    TitleView title = new TitleView();

    while(true){
      switch(nextView){
        case 0:
          {
            title = new TitleView();
            //タイトル画面を表示
            //bgm.setCycleCount(AudioClip.INDEFINITE);

            frame.add(title, BorderLayout.CENTER);
            frame.pack();
            frame.setVisible(true);
            if(bgm.isPlaying()==false)bgm.play();
            // Thread.sleep(12000000);
            while(true){
              if(bgm.isPlaying()==false)bgm.play();
              click_state_int = title.getClicked();
              System.out.printf("");
              if(click_state_int != 0){
                System.out.printf("");
                frame.remove(title);
                break;
              }
            }
          }
          if(click_state_int == 1) nextView = 1;
          else if(click_state_int == 2) nextView = 4;
          break;

        case 1:
          {
            StageSelectView stage = new StageSelectView();
            frame.add(stage, BorderLayout.CENTER);
            frame.setVisible(true);
            if(bgm.isPlaying()==false)bgm.play();
            while(true){
              if(bgm.isPlaying()==false)bgm.play();
              stage_num = stage.getStage();
              System.out.printf("");
              if(stage_num!=0){
                System.out.println("Stage " + stage_num);
                frame.remove(stage);

                break;
              }
            }
          }
          nextView = 2;
          break;

        case 2:
          {

            tm = new CourtModel(300,600,30,stage_num);
            GameView  tv = new GameView(tm);
            tv.setBlockModel(stage_num);
            frame.add(tv, BorderLayout.CENTER);
            frame.setVisible(true);
            if(bgm.isPlaying()==false)bgm.play();
            while(true){
              if(bgm.isPlaying()==false)bgm.play();
              state = tv.getState();
              System.out.printf("");
              if(state != 0){
                frame.remove(tv);
                break;
              }
            }
          }
          nextView = 3;
          break;

        case 3:
          {
            Finish finish = new Finish(state);
            frame.add(finish);
            frame.setVisible(true);
            if(bgm.isPlaying()==false)bgm.play();
            while(true){
              if(bgm.isPlaying()==false)bgm.play();
              click_state_int = finish.getClicked();
              System.out.printf("");
              if(click_state_int != 0){
                title = new TitleView();
                frame.remove(finish);
                break;
              }
            }
          }
          if(click_state_int == 1) nextView = 2;
          else if(click_state_int == 2) nextView = 0;
          break;

        case 4:
          StageMakeView stagemake = new StageMakeView();
          frame.add(stagemake, BorderLayout.CENTER);
          frame.pack();
          frame.setVisible(true);
          if(bgm.isPlaying()==false)bgm.play();
          while(true){
            if(bgm.isPlaying()==false)bgm.play();
            click_state = stagemake.getClicked();
            System.out.printf("");
            if(click_state){
              frame.remove(stagemake);
              break;
            }
          }
          nextView = 0;
          break;
      }
    }
  }
}

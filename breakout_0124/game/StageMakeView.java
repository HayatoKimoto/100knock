package game;
import java.awt.*;
import javax.swing.*;
import game.block.*;
import java.util.*;
import java.awt.event.*;
import java.awt.event.MouseListener;
import java.awt.event.MouseEvent;
import java.nio.charset.StandardCharsets;
import java.io.File;
import java.nio.file.Files;
import java.io.FileWriter;
import java.io.IOException;
import javax.imageio.ImageIO;
import java.awt.image.BufferedImage;
import java.nio.file.Path;
import java.nio.file.Paths;

public class StageMakeView extends JPanel implements MouseListener,ActionListener{
  ArrayList<Block> block_list = new ArrayList<Block>();
  private int type = 0,j=0,k=0; // 0:pen, 1:eraser
  private boolean clicked=false;//false:押してない,ture:押した
  private boolean result;
  File userpng = new File("user.png");
  public java.util.List<String> stage_text = new ArrayList<String>();

  public StageMakeView(){
    setBackground(Color.BLACK);
    setPreferredSize(new Dimension(301,601));
    setFocusable(true);

    try {
      stage_text = Files.readAllLines(Paths.get("user.txt"), StandardCharsets.UTF_8);
      System.out.println("print cournt");
      for(String s : stage_text) {
            System.out.println(s);
      }
    } catch (IOException e) {
        System.err.println( e);
    }

    //　ブロックをおける位置を点線で表示
    int max_y = 0;
    for(int y=15;max_y<10;y+=20){
      int max_x = 0;
      max_y++;
      for(int x=15;max_x<9;x+=30){
        Block block = new Block(0, 255, 255);
        block.setBlock(x,y);
        block.setCount(Integer.parseInt(stage_text.get(j).substring(k,k+1),10));
        block_list.add(block);
        max_x++;
        k++;
      }
      j++;
      k=0;
      addMouseListener(this);
    }



    // 画面下のアイコンを定義、表示
    {
      JPanel icons = new JPanel(new GridLayout(1,3));
      ImageIcon icon_save = new ImageIcon("game/icons/icons-1.png");
      ImageIcon icon_pen = new ImageIcon("game/icons/icons-2.png");
      ImageIcon icon_eraser = new ImageIcon("game/icons/icons-3.png");
      JButton bt_save = new JButton(icon_save);
      JButton bt_pen = new JButton(icon_pen);
      JButton bt_eraser = new JButton(icon_eraser);
      icons.setBackground(Color.BLACK);
      bt_save.setBorderPainted(false);
      bt_pen.setBorderPainted(false);
      bt_eraser.setBorderPainted(false);
      bt_save.addActionListener(this);
      bt_pen.addActionListener(this);
      bt_eraser.addActionListener(this);
      bt_save.setActionCommand("save");
      bt_pen.setActionCommand("pen");
      bt_eraser.setActionCommand("eraser");
      setLayout(new BorderLayout());
      icons.add(bt_eraser);
      icons.add(bt_pen);
      icons.add(bt_save);
      add(icons,BorderLayout.SOUTH);
    }
  }

  protected void paintComponent(Graphics g){
    super.paintComponent(g);
    System.out.println(type);
    if(type == 3){
      drawReal(g);
      System.out.println("real");
    }else{
      drawFrame(g);
      System.out.println("fra,e");
    }
  }

  public void mouseEntered(MouseEvent e){
  }

  public void mouseExited(MouseEvent e){
  }

  public void mousePressed(MouseEvent e){
  }

  public void mouseReleased(MouseEvent e){
  }

  public void mouseClicked(MouseEvent e){
    Point pt = e.getPoint();
    int x = e.getX();
    int y = e.getY();
    // System.out.println("x:" + x + "\ny:" + y);
    // どのブロックがクリックされたか調べる。
    // クリックされたブロックが見つかったらそのcountを1にする
    for(Block b: block_list){
      if(b.x < x && x < b.x+b.width && b.y < y && y < b.y+b.height){
        // System.out.println("blockfound");
        if(type == 0){
          b.setCount(1);
        }else if(type == 1){
          b.setCount(0);
        }
        break;
      }
      repaint();
    }
  }

  public void drawFrame(Graphics g){
    for(Block b: block_list){
      if(b.getCount() == 1){
        b.draw(g);
      }else{
        b.drawDotRect(g, 5, 0);
      }
    }
  }

  public void drawReal(Graphics g){
    for(Block b: block_list){
      if(b.getCount() == 1){
        b.draw(g);
      }else{
        b.drawDotRect(g, 5, 1);
      }
    }
  }

  public boolean getClicked(){
    return this.clicked;
  }

  public void actionPerformed(ActionEvent e){
    int i = 0;
    if(e.getActionCommand() != 	null){
      String cmd = e.getActionCommand();
      if(cmd.equals("save")){
        type = 3;
        Point point = new Point(this.getLocationOnScreen());
        Rectangle rect = new Rectangle(point.x+13 ,point.y+12, 274, 204);

        this.repaint();

        System.out.println("saved");
        try{
          File file = new File("user.txt");
          FileWriter filewriter = new FileWriter(file);

          for(Block b: block_list){
            i++;
            filewriter.write("" + b.getCount());
            if(i%9==0) filewriter.write("\n");
          }
          filewriter.close();
        }catch(IOException ex){
          System.out.println(ex);
        }
        try{
          Robot r = new Robot();
          BufferedImage img = r.createScreenCapture(rect);
          ImageIO.write(img,"jpg",userpng);

        } catch(Exception exo){
        }
        clicked = true;
      }
      if(cmd.equals("pen")){
        type = 0;
      }
      if(cmd.equals("eraser")){
        type = 1;
      }
    }
  }


}

package game.block;

import java.awt.*;
import java.awt.event.*;
import javax.swing.*;
import game.*;
import java.util.*;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.nio.file.Files;
import java.nio.charset.StandardCharsets;
import java.io.IOException;

public class BlockModel{
  private int x, y, max_x, max_y,i, r, g, b;
  private ArrayList<Block> block_list;
  public java.util.List<String> stage_text = new ArrayList<String>();
  public BlockModel(){
  }

  public void make_block(int a){
    i=-1;
    Block block;
    block_list=new ArrayList<Block>();
    String stage;
    int j=0, k=0, n;

    Path stage_text_path = Paths.get("stage1.txt");
    //
    switch(a){
      case 1:
      // Path file = Paths.get("/stage1.txt");
        stage_text_path = Paths.get("stage1.txt");
        break;
      case 2:
        stage_text_path = Paths.get("stage2.txt");
        break;
      case 3:
        stage_text_path = Paths.get("stage3.txt");
        break;
      case 4:
        stage_text_path = Paths.get("stage4.txt");
        break;
      case 5:
        stage_text_path = Paths.get("stage5.txt");
        break;
      case 6:
        stage_text_path = Paths.get("user.txt");
        break;
    }
    try {
      stage_text = Files.readAllLines(stage_text_path, StandardCharsets.UTF_8);
      System.out.println("print court");
      for(String s : stage_text) {
            System.out.println(s);
      }
    } catch (IOException e) {
        System.err.println( e);
    }
    r=255;
    g=150;
    b=180;
    max_y = 0;
    for(y=10;max_y<10;y+=20){
      max_x = 0;
      max_y++;
      r-=23;
      g-=2;
      b+=6;
    	for(x=15;max_x<9;x+=30){
        block = new Block(r, g, b);
        block.setBlock(x,y);
        block.setCount(Integer.parseInt(stage_text.get(j).substring(k,k+1),10));
        block_list.add(block);
        max_x++;
        k++;
      }
      j++;
	    k=0;
    }
  }

  public void draw(Graphics g){
    for(Block b: block_list){
      if(b.getCount() == 1){
        b.draw(g);
      }
    }
  }

  public int check(double ballet_x,double ballet_y){
    int i=0;
    for(Block b: block_list){
      if(b.checkhit(ballet_x,ballet_y)){
        return i;
      }
      i++;
    }
    return -1;
    //i = (int)(Math.random()*90);
  }

  public Block getBlock(int i){
    return block_list.get(i);
  }

  public void delete(int i){
    if(i==-1){
      return;
    }
    Block b = block_list.get(i);
    b.setCount(0);
  }

  public boolean clear_check(){
    for(Block b: block_list){
      if(b.getCount() == 1) return false;
    }
    return true;
  }

}

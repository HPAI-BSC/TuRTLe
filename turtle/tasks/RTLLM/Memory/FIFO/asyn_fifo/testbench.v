`timescale 1ns/1ns

module asyn_fifo_tb;

  reg wclk, rclk, wrstn, rrstn, winc, rinc;
  reg [7:0] wdata;
  wire wfull, rempty;
  wire [7:0] rdata;
  
  asyn_fifo #(.WIDTH(8), .DEPTH(16)) dut (
    .wclk(wclk),
    .rclk(rclk),
    .wrstn(wrstn),
    .rrstn(rrstn),
    .winc(winc),
    .rinc(rinc),
    .wdata(wdata),
    .wfull(wfull),
    .rempty(rempty),
    .rdata(rdata)
  );
  
  always #5 wclk = ~wclk;
  always #10 rclk = ~rclk;
  
  initial begin
    wclk = 0;
    rclk = 0;
    wrstn = 0;
    rrstn = 0;
    winc = 0;
    rinc = 0;
    wdata = 0;
  end

  // Apply reset and initialize FIFO
  initial begin
    wrstn = 0;
    rrstn = 0;
    #20;
    wrstn = 1;
    rrstn = 1;
    #10;
    winc = 1; // Enable write
    wdata = 8'hAA; // Write data
    #10;
    winc = 0; // Disable write
    #500;
    rinc = 1;
    #500;
    #10;
    $finish;
  end

  integer error = 0;
  reg [31:0] data1 [0:50]; // wfull data
  reg [31:0] data2 [0:50]; // rempty data
  reg [31:0] data3 [0:50]; // tdata data
  integer i = 0;

  initial begin
    #550;
    // Hardcode wfull.txt data into data1
    data1[0] = 1; data1[1] = 1; data1[2] = 1;
    for (integer j = 3; j < 48; j++) data1[j] = 0;

    // Hardcode rempty.txt data into data2
    for (integer j = 0; j < 48; j++)
      data2[j] = (j == 32 || j == 33) ? 1 : 0;

    // Hardcode tdata.txt data into data3
    data3[0] = 8'h01; data3[1] = 8'h01;
    data3[2] = 8'hab; data3[3] = 8'hab;
    data3[4] = 8'hac; data3[5] = 8'hac;
    data3[6] = 8'had; data3[7] = 8'had;
    data3[8] = 8'hae; data3[9] = 8'hae;
    data3[10] = 8'haf; data3[11] = 8'haf;
    data3[12] = 8'hb0; data3[13] = 8'hb0;
    data3[14] = 8'hb1; data3[15] = 8'hb1;
    data3[16] = 8'hb2; data3[17] = 8'hb2;
    data3[18] = 8'hb3; data3[19] = 8'hb3;
    data3[20] = 8'hb4; data3[21] = 8'hb4;
    data3[22] = 8'hb5; data3[23] = 8'hb5;
    data3[24] = 8'hb6; data3[25] = 8'hb6;
    data3[26] = 8'hb7; data3[27] = 8'hb7;
    data3[28] = 8'hb8; data3[29] = 8'hb8;
    data3[30] = 8'hb9; data3[31] = 8'hb9;
    data3[32] = 8'h01; data3[33] = 8'h01;
    data3[34] = 8'h01; data3[35] = 8'h01;
    data3[36] = 8'hab; data3[37] = 8'hab;
    data3[38] = 8'hac; data3[39] = 8'hac;
    data3[40] = 8'had; data3[41] = 8'had;
    data3[42] = 8'hae; data3[43] = 8'hae;
    data3[44] = 8'haf; data3[45] = 8'haf;
    data3[46] = 8'hb0; data3[47] = 8'hb0;

    repeat(48) begin
      #10;
      error = (wfull == data1[i] && rempty == data2[i] && rdata == data3[i]) ? error : error + 1;
      i++;
    end

    if (error == 0) $display("===========Your Design Passed===========");
    else $display("===========Error===========");
  end

  initial begin
    repeat (17) begin
      #20;
      if (wfull) break;
      winc = 1;
      wdata = wdata + 1;
      #10;
      winc = 0;
    end
  end
  
endmodule

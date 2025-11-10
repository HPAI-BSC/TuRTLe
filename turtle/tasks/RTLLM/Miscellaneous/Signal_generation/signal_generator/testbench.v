module tb_signal_generator;
reg				clk,rst_n;
wire[4:0]		wave;

signal_generator uut(
				.clk(clk),
				.rst_n(rst_n),
				.wave(wave)
				);

reg[31:0]reference[0:99];

integer i = 0;
integer error = 0;

initial begin
    integer j; // Loop variable declaration

    // Initialize reference array with hardcoded values
    // First segment: 0x00 to 0x1f (0-31)
    for (j = 0; j < 32; j = j + 1) begin
        reference[j] = j;
    end
    // Second segment: 0x1f to 0x00 (32-63)
    for (j = 0; j < 32; j = j + 1) begin
        reference[32 + j] = 31 - j;
    end
    // Third segment: 0x00 to 0x1f (64-95)
    for (j = 0; j < 32; j = j + 1) begin
        reference[64 + j] = j;
    end
    // Fourth segment: 0x1f to 0x1c (96-99)
    for (j = 0; j < 4; j = j + 1) begin
        reference[96 + j] = 31 - j;
    end

    clk = 0;
    rst_n = 0;
    #10;
    rst_n = 1;

    repeat(100) begin
        error = (wave == reference[i]) ? error : error + 1;
        #10;
        i = i + 1;
    end

    if (error == 0) begin
        $display("===========Your Design Passed===========");
    end else begin
        $display("===========Error===========");
    end
    $finish;
end
 
always #5 clk = ~clk;
 
endmodule

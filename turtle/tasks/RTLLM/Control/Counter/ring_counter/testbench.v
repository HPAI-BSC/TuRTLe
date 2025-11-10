module testbench;

    reg clk;
    reg reset;
    wire [7:0] out;

    ring_counter ring_counter_inst (
        .clk(clk),
        .reset(reset),
        .out(out)
    );

    always begin
        #5 clk <= ~clk;
    end

    reg [3:0] i;
    reg [7:0] data [0:9];  // Fix

    initial begin
        // Fix: Icarus Verilog does not support initialization with {}
        data[0] = 8'b00000001;
        data[1] = 8'b00000001;
        data[2] = 8'b00000010;
        data[3] = 8'b00000100;
        data[4] = 8'b00001000;
        data[5] = 8'b00010000;
        data[6] = 8'b00100000;
        data[7] = 8'b01000000;
        data[8] = 8'b10000000;
        data[9] = 8'b00000001;
    end

    initial begin
        clk = 0;
        reset = 1;
        i = 0;
        #10 reset = 0;
    end

    always @(posedge clk) begin
        if (out !== data[i]) begin
            $display("Failed at i=%d, out=%b, expected=%b", i, out, data[i]);
        end
        i <= i + 1;
    end

    always @(posedge clk) begin
        if (i == 9) begin
            $display("=========== Your Design Passed ===========");
            $finish;
        end
    end

    initial begin
        #100 $finish;
    end

endmodule

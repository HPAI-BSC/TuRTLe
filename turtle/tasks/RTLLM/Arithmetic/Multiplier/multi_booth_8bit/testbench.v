`timescale 1ns/1ns
`define width 8

module booth4_mul_tb () ;
    reg signed [`width-1:0] a, b;
    reg             clk, reset;

    wire signed [2*`width-1:0] p;
    wire           rdy;

    integer total, err;
    integer i, numtests;

    // Test data arrays
    reg signed [`width-1:0] a_tests [0:9];
    reg signed [`width-1:0] b_tests [0:9];

    wire signed [2*`width-1:0] ans = a*b;

    multi_booth_8bit dut( .clk(clk),
        .reset(reset),
        .a(a),
        .b(b),
        .p(p),
        .rdy(rdy));

    // Set up 10ns clock
    always #5 clk = ~clk;

    task apply_and_check;
        input [`width-1:0] ain;
        input [`width-1:0] bin;
        begin
            a = ain;
            b = bin;
            reset = 1;
            @(posedge clk);
            #1 reset = 0;

            while (rdy == 0) begin
                @(posedge clk);
            end
            if (p != ans) begin
                err = err + 1;
            end
            total = total + 1;
        end
    endtask

    initial begin
        clk = 1;
        total = 0;
        err = 0;

        // Hardcoded test data
        numtests = 10;
        a_tests[0] = 5;   b_tests[0] = 5;
        a_tests[1] = 2;   b_tests[1] = 3;
        a_tests[2] = 10;  b_tests[2] = 1;
        a_tests[3] = 10;  b_tests[3] = 2;
        a_tests[4] = 20;  b_tests[4] = 20;
        a_tests[5] = -128;b_tests[5] = 2;
        a_tests[6] = 10;  b_tests[6] = -128;
        a_tests[7] = -1;  b_tests[7] = -1;
        a_tests[8] = 10;  b_tests[8] = 0;
        a_tests[9] = 0;   b_tests[9] = 2;

        for (i = 0; i < numtests; i = i + 1) begin
            apply_and_check(a_tests[i], b_tests[i]);
        end

        if (err > 0) begin
            $display("=========== Failed ===========");
        end else begin
            $display("===========Your Design Passed===========");
        end
        $finish;
    end

endmodule

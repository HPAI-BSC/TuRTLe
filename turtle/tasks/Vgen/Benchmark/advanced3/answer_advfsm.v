// Finite State Machine to detect the sequence "101" on input signal x.
module adv_fsm (
    input clk,
    input reset,
    input x,
    output reg z
);

    // Define states using 2-bit encoding
    reg [1:0] present_state, next_state;
    parameter IDLE = 2'b00, S1 = 2'b01, S10 = 2'b10, S101 = 2'b11;

    // Output signal z is asserted when present_state is S101
    always @ (posedge clk or posedge reset) begin
        if (reset) begin
            present_state <= IDLE; // Reset to IDLE state
            z <= 0;                // Clear output
        end else begin
            present_state <= next_state; // Update state
            if (present_state == S101) begin
                z <= 1; // Assert output for one clock cycle
            end else begin
                z <= (next_state == S101); // Deassert output otherwise
            end
        end
    end

    // Next state logic
    always @ (*) begin
        case (present_state)
            IDLE: begin
                if (x) next_state = S1;
                else next_state = IDLE;
            end
            S1: begin
                if (x) next_state = IDLE;
                else next_state = S10;
            end
            S10: begin
                if (x) next_state = S101;
                else next_state = IDLE;
            end
            S101: begin
                next_state = IDLE; // Reset after detecting "101"
            end
            default: next_state = IDLE; // Default case
        endcase
    end

endmodule

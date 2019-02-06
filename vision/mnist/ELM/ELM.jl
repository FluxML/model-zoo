module Elm
    export elmtrain, elmpredict
    
    #keeping weights as global variables
    win = []
    wout = []
    """
        function elmtrain
        input: 1) n * m train set; where m = #samples, n = #features
               2) c * m labels; where m = #samples, c = #classes
        action: generates win and wout
    """
    function elmtrain(train_x, train_y, hidden_units)
        #fixing layer parameters
        input_units = size(train_x,1)

        #Generating win using randn
        win = randn(hidden_units, input_units)
    
        #first step of forward propagation
        hidden_layer = win*train_x
    
        #applying ReLu
        hidden_layer = hidden_layer .* (hidden_layer .> 0)
    
        #calculating wout as (H^T*H)^(-1)*(H^T*Y)
        #wout = inv(transpose(hidden_layer)*hidden_layer)*(transpose(hidden_layer)*train_y)
        wout = (train_y*transpose(hidden_layer))*inv(hidden_layer*transpose(hidden_layer))
        global win = win
        global wout = wout
        end

    """
        function elmpredict
        input: m * n test set; where m = #samples, n = #features
        action: forward propagation and predictions
    """
    function elmpredict(test_x)
        
        
        #first step of forward propagation
        hidden_layer = win*test_x
        #applying ReLu
        hidden_layer = hidden_layer .* (hidden_layer .> 0)
        #second step of forward propagation
        output_layer = wout*hidden_layer
    
        #making predictions
        _, indices = findmax(output_layer, dims = 1)
        predictions = zeros(size(output_layer))
    
        for i in 1:size(test_x,2)
            predictions[indices[i]] = 1
        end
        return predictions
    end
end
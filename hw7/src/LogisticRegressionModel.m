classdef LogisticRegressionModel < handle
    properties 
        % 
        C
        % eta
        learning_rate
        % gradient relative condition
        epsilon
        % CG xi
        xi
        % weight: 
        w
        % training method = "gradient" or "newton"
        training_method
        % 
        max_iterations
        %
        max_alpha_iterations
        % 
        loss
        % model file name
        name 
    end
    methods
        % Constructor
        function obj = LogisticRegressionModel(C, learning_rate, epsilon, xi, training_method, max_iterations)
            obj.C = C;
            obj.learning_rate = learning_rate;
            obj.epsilon = epsilon;
            obj.xi = xi;
            obj.training_method = training_method;
            obj.max_iterations = max_iterations;
            
            obj.w = 0;
            obj.loss = [];
            name = sprintf('tryC%f,lr%f,epsilon%f,xi%f,%s,', C, learning_rate, epsilon, xi, training_method);
            obj.name = name;

        end 
        % Dump model
        % Load model
        % Train
        function fit(obj, instance_matrix, label_vector)
            % Create weight
            dimensions = size(instance_matrix);
            feature_dim = dimensions(2);
            obj.w = zeros(1, feature_dim);
            % Choose method
            if strcmp(obj.training_method, 'gradient')
                if abs(obj.C - 0.01) < 1e-5
                    obj.max_alpha_iterations = 100;
                elseif abs(obj.C - 0.1) < 1e-5
                    obj.max_alpha_iterations = 100; 
                end
                obj.gradient_method_train(label_vector, instance_matrix);
                
                                  
            elseif strcmp(obj.training_method, 'newton')
                if abs(obj.C - 0.01) < 1e-5
                    obj.max_alpha_iterations = 10;
                elseif abs(obj.C - 0.1) < 1e-5
                    obj.max_alpha_iterations = 10; 
                end
                obj.netwon_method_train(label_vector, instance_matrix);
            end
        end
        % 
        function label_vector = predict(obj, instance_matrix)
            %
            probability = 1 ./ (1 + exp(-obj.w * transpose(instance_matrix)));
            % 1
            positive_index = (probability >= 0.5);
            % -1
            negative_index = (probability < 0.5);
            % 
            label_vector = probability;
            label_vector(positive_index) = 1;
            label_vector(negative_index) = -1;
            % (1, N) -> (N, 1)
            label_vector = transpose(label_vector);
        end
        % load model
        function load(obj)
            obj.w = dlmread(fullfile('../parameter', strcat(obj.name, 'w')));
            obj.w = transpose(obj.w);
        end
        % 
        function gradient_method_train(obj, y, x)
            disp('======================= Using gradient descent method ===============');
            % Save w_x
            % An (1, N) matrix: [w x1, w x2, w x3 ... ]
            w = obj.w;
            w_x = obj.w * transpose(x);
            f = @obj.f;
            
            first_gradient_norm = norm(obj.gradient(w, y, x, w_x));
            % Training loop
            for k = 1:obj.max_iterations
                % compute gradient: g == ()
                g = obj.gradient(w, y, x, w_x);
                s = -g;
                % Backtracking Line Search
                alpha_ = 1;
                now_f_w = f(w, y, x, w_x, 0);
                obj.loss = [obj.loss now_f_w];
                % Record training loss
                fprintf('Iterations: %d Loss: %.18f\n', k, now_f_w);
                % Dump Loss
                dlmwrite(fullfile('../loss', strcat(obj.name, 'loss')), obj.loss, 'delimiter', '\t', 'precision', 16)
                
                % Backtracking line search
                g_s_product =  g * transpose(s);
                alpha_iterations = 0;
                while(true)
                    alpha_iterations = alpha_iterations + 1;
                    lhs = f(w + alpha_ * s, y, x, w_x, alpha_ * s);
                    rhs = now_f_w + obj.learning_rate * alpha_ * g_s_product;
                    if(isinf(lhs))
                        alpha_ = alpha_ / 2;
                        continue
                    elseif(lhs <= rhs || alpha_iterations > obj.max_alpha_iterations)
                        fprintf('alpha_iterations: %d\n', alpha_iterations)
                        break 
                    end   
                    alpha_ = alpha_ / 2;
                end
                % Update weight
                w = w + alpha_ * s;
                % Update w_x
                w_x = w_x + alpha_ * s * transpose(x);
                % Check stopping condition
                fprintf('Norm Ratio: %.30f\n', norm(g) / first_gradient_norm)
                if norm(g) <= obj.epsilon * first_gradient_norm
                    break
                end      
            end
            % Save model to itself
            obj.w = w;
            now_f_w = f(w, y, x, w_x, 0);
            obj.loss = [obj.loss now_f_w];
            dlmwrite(fullfile('../loss', strcat(obj.name, 'loss')), obj.loss, 'delimiter', '\t', 'precision', 16);
        end
        %
        function netwon_method_train(obj, y, x)
            disp('======================= Using newton method ===============');
            w = obj.w;
            w_x = obj.w * transpose(x);
            f = @obj.f;
            first_gradient_norm = norm(obj.gradient(w, y, x, w_x));
            % Training Loop
            for k = 1:obj.max_iterations
                g = obj.gradient(w, y, x, w_x);
                % Get Netwon Direction
                s = obj.conjugate_gradient_method(0, -g, w, y, x, w_x);
                % 
                alpha_ = 1;
                now_f_w = f(w, y, x, w_x, 0);
                % Record Loss
                obj.loss = [obj.loss now_f_w];
                % Print Loss
                fprintf('Iterations: %d Loss: %.18f\n', k, now_f_w);
                % Dump Loss
                dlmwrite(fullfile('../loss', strcat(obj.name, 'loss')), obj.loss, 'delimiter', '\t', 'precision', 16)
                %
                g_s_product = g * transpose(s); % (1, feature_sim) 
                alpha_iterations = 0;
                while(true)
                    alpha_iterations = alpha_iterations + 1;
                    lhs = f(w + alpha_ * s, y, x, w_x, alpha_ * s);
                    rhs = now_f_w + obj.learning_rate * alpha_ * g_s_product;
                    if(isinf(lhs))
                        alpha_ = alpha_ / 2;
                        continue
                    elseif(lhs <= rhs || alpha_iterations > obj.max_alpha_iterations)
                        fprintf('alpha_iterations: %d\n', alpha_iterations)
                        break 
                    end   
                    alpha_ = alpha_ / 2;
                end
                % Update weight
                w = w + alpha_ * s;
                % Update w_x
                w_x = w_x + alpha_ * s * transpose(x);
                % Stopping condition
                fprintf('Norm Ratio: %.30f\n', norm(g) / first_gradient_norm)
                if norm(g) <= obj.epsilon * first_gradient_norm
                    break
                end   
            end
            % Save to model
            obj.w = w;
            now_f_w = f(w, y, x, w_x, 0);
            obj.loss = [obj.loss now_f_w];
            dlmwrite(fullfile('../loss', strcat(obj.name, 'loss')), obj.loss, 'delimiter', '\t', 'precision', 16)
        end
        % Conjugate Gradient Method
        function s = conjugate_gradient_method(obj, s_0, r_0, w, y, x, w_x)
            %{
                r_0: (1, feature_dim)
                A: Here, is hessian of f(w)
            %}
            s = s_0;
            r_i = r_0;
            r_i_add_1 = 0;
            d = r_0;
            norm_r_0 = norm(r_0);
            clear r_0
            % CG Loop
            while norm(r_i) > obj.xi * norm_r_0
                % Compute hessian multiply d
                hessian_multiply_d = obj.hessian_vector_product(w, y, x, w_x, d);
                %
                alpha_ = (norm(r_i) ^ 2) / (d * transpose(hessian_multiply_d));
                % Update s
                s = s + alpha_ * d;
                % 
                r_i_add_1 = r_i - alpha_ * hessian_multiply_d;
                % 
                beta_ = (norm(r_i_add_1) ^ 2) / (norm(r_i) ^ 2);
                % 
                d = r_i_add_1 + beta_ * d;
                % Update r_i
                r_i = r_i_add_1;
                r_i_add_1 = 0;
                % Free memory
                clear hessian_multiply_d
            end 
        end
        
        function value = f(obj, w, y, x, w_x, direction)
            % w is the weight now. shape == (1, feature_dim)
            % w_x is last step's value. shape == (1, N)
            % y. shape == (N, 1)
            % x. shape == (N, feature_dim)
            % direction. shape == (1, feature_dim)
            value = 0;
            % Left part: (1, feature_dim) * (feature_dim, 1)
            value = value + (1/2) * w * transpose(w);
            % Right part
            if all(direction == 0)
                exponent = -1 * transpose(y) .* (w_x);
            else
                exponent = -1 * transpose(y) .* (w_x + direction * transpose(x));
            end
            value = value + obj.C * sum(log(1 + exp(exponent)));
        end
        % Compute gradient
        function g = gradient(obj, w, y, x, w_x)
            % Right part: (N, 1)^T .* (1, N) * (N, feature_dim)
            g = w + obj.C * ((transpose(y) .* ((1 ./ (1 + exp(-1 * transpose(y) .* w_x))) - 1)) * x);
        end
        % Hessian vector product
        function h_s = hessian_vector_product(obj, w, y, x, w_x, direction)
            % direction: (1, feature_dim)
            % D with shape (1, N)
            tmp_y_w_x = exp(-transpose(y) .* w_x);
            D_vector_form = tmp_y_w_x ./ ((1 + tmp_y_w_x) .^ 2);
            %{
                x * transpose(direction): (N, 1)
                D_vector_form: (1, N)
                tranpose(x): (N, feature_dim)
                return h_s: (1, feature_dim)
            %}
            h_s = transpose(direction) + obj.C * (transpose(x) * (transpose(D_vector_form) .* (x * transpose(direction))));
            h_s = transpose(h_s);
            clear tmp_y_w_x
            clear D_vector_form
            clear w y x w_x direction
        end
    end
end
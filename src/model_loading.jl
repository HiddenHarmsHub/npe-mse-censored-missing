function load_model_nbe(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m, models_path)
    mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    if isfile(joinpath(models_path, mdl_str))
        model = BSON.load(joinpath(models_path, mdl_str))
        return model[:estimator]
    else
        error("Model file $(mdl_str) not found in $(models_path).")
    end
end

function load_model_npe(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m, models_path; encoding_dim = 128)
    mdl_str = "model_$(n_lists)_$(width)_$(n_hidden)_$(encoding_dim)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
    if isfile(joinpath(models_path, mdl_str))
        model = BSON.load(joinpath(models_path, mdl_str))
        return model[:estimator]
    else
        error("Model file $(mdl_str) not found in $(models_path).")
    end
end

function load_model_nbe(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m, models_path, ci::Bool)
    if !ci
        load_model_nbe(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m, models_path)
    else
        mdl_str = "model_ci_$(n_lists)_$(width)_$(n_hidden)_$(train_size)_$(censoring_lower)_$(censoring_upper)_$m.bson"
        if isfile(joinpath(models_path, mdl_str))
            model = BSON.load(joinpath(models_path, mdl_str))
            return model[:ci_estimator]
        else
            error("Model file $(mdl_str) not found in $(models_path).")
        end
    end 
end

function load_model_nbe(mdl_str, models_path)
    if isfile(joinpath(models_path, mdl_str))
        model = BSON.load(joinpath(models_path, mdl_str))
        if occursin("ci_", mdl_str)
            return model[:ci_estimator]
        end
        return model[:estimator]
    else
        error("Model file $(mdl_str) not found in $(models_path).")
    end
end

function load_model_nbe(; 
    n_lists, 
    width, 
    n_hidden, 
    train_size, 
    censoring_lower,
    censoring_upper,
    m,
    ci = false, 
    models_path = joinpath("output", "models_nbe")
)
    load_model_nbe(n_lists, width, n_hidden, train_size, censoring_lower, censoring_upper, m, models_path, ci)
end
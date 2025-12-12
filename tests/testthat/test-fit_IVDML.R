test_that("fit_IVDML works when X == NULL", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- rnorm(10)
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "gam"))
})



test_that("fit_IVDML raises error when input lengths do not match", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- rnorm(10)
  X <- rnorm(10)
  expect_error(fit_IVDML(Y = Y[1:9], D = D, Z = Z, X = X, ml_method = "gam"))
  expect_error(fit_IVDML(Y = Y, D = D[1:9], Z = Z, X = X, ml_method = "gam"))
  expect_error(fit_IVDML(Y = Y, D = D, Z = Z[1:9], X = X, ml_method = "gam"))
  expect_error(fit_IVDML(Y = Y, D = D, Z = Z, X = X[1:9], ml_method = "gam"))
})

test_that("fit_IVDML raises warning if A is NULL but A_deterministic_X is FALSE", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- rnorm(10)
  X <- rnorm(10)
  expect_warning(fit_IVDML(Y = Y, D = D, Z = Z, X = X, A_deterministic_X = FALSE, ml_method = "gam"))
})

test_that("fit_IVDML works without warning when A is specified but A_deterministic_X is FALSE", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- rnorm(10)
  X <- rnorm(10)
  A <- rnorm(10)
  expect_no_warning(fit_IVDML(Y = Y, D = D, Z = Z, X = X, A = A, A_deterministic_X = FALSE, ml_method = "gam"))
})

test_that("fit_IVDML raises warning if dim(A)>1", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- rnorm(10)
  X <- rnorm(10)
  A <- cbind(rnorm(10), rnorm(10))
  expect_warning(fit_IVDML(Y = Y, D = D, Z = Z, X = X, A = A, A_deterministic_X = FALSE, ml_method = "gam"))
})

test_that("fit_IVDML works when Z or X are given as dataframes.", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- data.frame(cbind(rnorm(10), rnorm(10)))
  X <- data.frame(cbind(rnorm(10), rnorm(10)))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = X, ml_method = "gam"))
})

test_that("fit_IVDML raises error when ml_method is not available", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- c(rnorm(10), rnorm(10))
  X <- c(rnorm(10), rnorm(10))
  expect_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "gaaaaaaaaaaaaaam"))
})

test_that("fit_IVDML raises error when iv_method is not available", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- rnorm(10)
  X <- rnorm(10)
  expect_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "gam", iv_method = "MLIV"))
})

test_that("fit_IVDML works when ml_par are set globally.", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- cbind(rnorm(10), rnorm(10))
  X <- cbind(rnorm(10), rnorm(10))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "gam", ml_par = list(ind_lin_X = 1, ind_lin_Z = c(1, 2))))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "xgboost", ml_par = list(eta = c(0.1, 0.2), max_depth = c(2, 3, 4))))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "randomForest", ml_par = list(num.trees = 20, min.node.size = 3)))
})


test_that("fit_IVDML works when ml_par is set differently for the different nuisance functions.", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- cbind(rnorm(10), rnorm(10))
  X <- cbind(rnorm(10), rnorm(10))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "gam", ml_par = list(ml_par_D_XZ = list(ind_lin_X = 1, ind_lin_Z = c(1, 2)), ml_par_Y_X = list(ind_lin_X = 2))))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "xgboost", ml_par = list(ml_par_D_XZ =list(eta = c(0.1, 0.2), max_depth = c(2, 3, 4)), ml_par_f_X = list(eta = c(0.1, 0.5)))))
  expect_no_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "randomForest", ml_par = list(ml_par_Z_X = list(num.trees = 20, min.node.size = 3))))
})

test_that("fit_IVDML raises error when ml_par is set inconsistently.", {
  set.seed(1)
  Y <- rnorm(10)
  D <- rnorm(10)
  Z <- cbind(rnorm(10), rnorm(10))
  X <- cbind(rnorm(10), rnorm(10))
  expect_error(fit_IVDML(Y = Y, D = D, Z = Z, X = NULL, ml_method = "gam", ml_par = list(ml_par_D_XZ = list(ind_lin_X = 1, ind_lin_Z = c(1, 2)), ind_lin_X = 1)))
})


test_that("ml methods work without specifying ml_par", {
  set.seed(1)
  n <- 20
  X <- cbind(rnorm(n), rnorm(n))
  Y <- abs(X[, 1]) + 0.5 * rnorm(n)
  Xnew <- cbind(rnorm(n), rnorm(n))
  fit_par_gam <- tune_gam(Y, X, ml_par = list())
  fit_par_xgb <- tune_xgboost(Y, X, ml_par = list())
  fit_par_randomForest <- tune_randomForest(Y, X, ml_par = list())
  fitted_gam <- fit_gam(Y, X, fit_par_gam)
  fitted_xgb <- fit_xgboost(Y, X, fit_par_xgb)
  fitted_randomForest <- fit_randomForest(Y, X, fit_par_randomForest)
  pred_gam <- predict_gam(Xnew, fitted_gam)
  pred_xgb <- predict_xgboost(Xnew, fitted_xgb)
  pred_randomForest <- predict_randomForest(Xnew, fitted_randomForest)
})


test_that("ml_methods work with specifying ml_par", {
  set.seed(1)
  n <- 20
  X <- cbind(rnorm(n), rnorm(n))
  Y <- abs(X[, 1]) + 0.5 * rnorm(n)
  Xnew <- cbind(rnorm(n), rnorm(n))
  ml_par_xgb <- list(max_nrounds = 10,
                     eta = c(0.11, 0.21, 0.31),
                     max_depth = c(1, 2, 3, 4, 5, 6, 7),
                     early_stopping_rounds = 8,
                     k_cv = 8)
  ml_par_randomForest <- list(num.trees = 10,
                              num_mtry = 5,
                              num_min.node.size = 10)
  fit_par_xgb <- tune_xgboost(Y, X, ml_par = ml_par_xgb)
  fit_par_randomForest <- tune_randomForest(Y, X, ml_par = ml_par_randomForest)
  fitted_xgb <- fit_xgboost(Y, X, fit_par_xgb)
  fitted_randomForest <- fit_randomForest(Y, X, fit_par_randomForest)
  pred_xgb <- predict_xgboost(Xnew, fitted_xgb)
  pred_randomForest <- predict_randomForest(Xnew, fitted_randomForest)
})


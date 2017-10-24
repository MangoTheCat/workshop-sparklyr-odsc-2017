library(sparklyr)
library(dplyr)

config <- spark_config()
# config$sparklyr.cores.max <- 64
# config$spark.executor.cores <- 32
# config$spark.executor.memory <- "16G"

sc <- spark_connect(master = "spark://ip-*************.eu-west-1.compute.internal:7077",
                    app_name = "doug",
                    config = config)

flights <- spark_read_parquet(sc, 
                              name = "flights08",
                              path = "/workshop/2008.parquet/")


# Import other data stored as CSV files
airports <- spark_read_csv(sc, 
                           name = "airports",
                           path = "/workshop/airports.csv")

carriers <- spark_read_csv(sc, name = "carriers", 
                           path = "/workshop/carriers.csv")

planes <- spark_read_csv(sc, name = "planes",
                         path = "/workshop/plane-data.csv")


# dplyr -------------------------------------------------------------------


## The Basics of dplyr
day15 <- filter(flights, DayofMonth == 15) # Return on day 15 data

small15 <- select(day15, Year, DayOfWeek, Origin) # Return only selected columns

mutate(day15, Date = paste(Year, Month, DayofMonth, sep = "-")) # create new column

arrange(day15, Month, DepDelay) # re-order the data

summarise(day15, mean(DepDelay)) # generate a numeric summary


# Works with the pipe as usual
flights %>%
  filter(DayofMonth == 15) %>%
  select(Year, DayOfWeek, Origin) %>%
  collect()

# Check the names of the tables available in Spark
db_list_tables(sc)

# Show the Spark SQL query generated
show_query(day15)

# Bring all of the results from the query into R
collect(day15)

# Join data using the usual dplyr join functions
friday <- flights %>%
  filter(DayOfWeek == 5) %>%
  left_join(airports, by = c(Dest = "iata")) %>%
  filter(Origin %in% c("SFO", "BOS"))

# Create a table in Spark that contains the output of running the query
sdf_register(friday, "friday")

# Use the tbl function to create an R object reference to the Spark data frame
friday <- tbl(sc, "friday")


# MLlib -------------------------------------------------------------------


## Machine Learning with sparklyr

copy_to(sc, iris)

iris_tbl <- tbl(sc, "iris")

# Use the binarizer to create a binary column (yes/no response)
iris_tbl <- ft_binarizer(iris_tbl, input.col = "Sepal_Length",
                         output.col = "SL_Group", 
                         threshold = 6)

# Use the binarizer to create an indicator variable (flights more than 30 minutes late departing are late)
flights %>%
  mutate(DepDelayDb = as.numeric(DepDelay)) %>%
  sdf_mutate(Late = ft_binarizer(DepDelayDb, 30)) %>%
  select(starts_with("Dep"), Late) %>%
  filter(!is.na(Late))

# Create dummy variables from R factors
ml_create_dummy_variables(iris_tbl, "Species") %>%
  select(contains("Species"))


# Create n discrete buckets
iris_tbl %>%
  sdf_mutate(SL_Buckets = ft_quantile_discretizer(Sepal_Length, n.buckets = 5))


# Create buckets with particular breaks
iris_tbl %>%
  sdf_mutate(SL_Split = ft_bucketizer(Sepal_Length, splits = 1:6))


# Using the binarizer to define morning flights
friday %>% 
  mutate(DepTimeDb = as.numeric(DepTime)) %>%
  sdf_mutate(AM = ft_binarizer(DepTimeDb, threshold = 1200)) %>%
  select(DepTime, AM)


# Partition the data for analysis
fl_part <- friday %>%
  select(ArrDelay, DepTime) %>%
  na.omit() %>%
  sdf_partition(train = 0.8, test = 0.2, seed = 321)


# Use sparklyr to fit linear regression on training data
model <- ml_linear_regression(fl_part$train, ArrDelay ~ DepTime)


# Usual model functions apply
class(model)
coef(model)
summary(model)

broom::glance(model)
broom::tidy(model)

# Create a new data set for model and partition
fl_part <- friday %>%
  select(Cancelled, DepTime, state) %>%
  na.omit() %>%
  sdf_partition(train = 0.8, test = 0.2, seed = 321)

# Fit a logistic regression
model_class <- ml_logistic_regression(fl_part$train, Cancelled ~ DepTime + state)
summary(model_class)

# Predict from the results
pred <- sdf_predict(model_class, newdata = fl_part$test)

# Generate metrics from model fit
ml_binary_classification_eval(pred, 
                              label = "Cancelled",
                              score = "probability",
                              metric = "areaUnderROC")


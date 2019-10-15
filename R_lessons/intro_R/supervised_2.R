----------------------
# Set up environment                   ---
#----------------------------------------
# clear global environment
rm(list = ls())

setwd("C:/Users/kevin/Dropbox/TAD_2019/R_practice_staging")

# load required libraries
library(quanteda)
library(readtext)
library(dplyr)

#----------------------------------------
# 1 Supervised Learning: Naive Bayes     ---
#----------------------------------------
#source of data: https://www.kaggle.com/rmisra/news-category-dataset#News_Category_Dataset_v2.json
#library(rjson)
#json_file <- "/Users/pedrorodriguez/Downloads/News_Category_Dataset_v2.json"
#con = file(json_file, "r") 
#input <- readLines(con, -1L) 
#news_data <- lapply(X=input,fromJSON)
#news_data <- lapply(news_data, function(x) as_tibble(t(unlist(x))))
#news_data <- do.call(rbind, news_data)
#saveRDS(news_data, "~/Dropbox/NYU/Teaching/Text as Data/TaD-2018/W6_02_27_18/news_data.rds")

# load data
news_data <- readRDS("news_data.rds")

## BIG Dataset!

# subset data and keep relevant variables
#filter-- restricts dataset, dropping rows

news_samp <- filter(news_data, category %in% c("CRIME", "SPORTS"))
  
##select --- keeps only named columns

news_samp1 <- select(news_samp, headline, category)


##setNames --- rename the variables in our new dataset

news_samp2<-setNames(object = news_samp1, nm = c("text", "class"))


# get a sense of how the text looks
dim(news_samp2)
head(news_samp2$text[news_samp2$class == "CRIME"])
head(news_samp2$text[news_samp2$class == "SPORTS"])

# some pre-processing (the rest will let dfm do)
news_samp2$text <- gsub(pattern = "'", "", news_samp2$text)  # replace apostrophes
head(news_samp2$text[news_samp2$class == "SPORTS"])

# what's the distribution of classes?
prop.table(table(news_samp2$class))

# split sample into training & test sets
set.seed(1984L)

###what does this do??

prop_train <- 0.8

###need to create a training set that's 80% of the data
ids <- 1:nrow(news_samp2)

ids_train <- sample(ids, ceiling(prop_train*length(ids)), replace = FALSE)

## the - sign is "not"
ids_test <- ids[-ids_train]
train_set <- news_samp2[ids_train,]
test_set <- news_samp2[ids_test,]

# get dfm for each set
train_dfm <- dfm(train_set$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english"))
test_dfm <- dfm(test_set$text, stem = TRUE, remove_punct = TRUE, remove = stopwords("english"))

# how does this look?
as.matrix(train_dfm)[1:5,1:5]

# match test set dfm to train set dfm features
test_dfm <- dfm_match(test_dfm, features = featnames(train_dfm))

# w/o smoothing ----------------

# train model on the training set
nb_model <- textmodel_nb(train_dfm, train_set$class, smooth = 0, prior = "uniform")

# evaluate on test set
predicted_class <- predict(nb_model, newdata = test_dfm)

# baseline --- This is important, to see how much our model beats a model that just picks the modal class 
baseline_acc <- max(prop.table(table(test_set$class)))

# get confusion matrix
cmat <- table(test_set$class, predicted_class)
nb_acc <- sum(diag(cmat))/sum(cmat) # accuracy = (TP + TN) / (TP + FP + TN + FN)
nb_recall <- cmat[2,2]/sum(cmat[2,]) # recall = TP / (TP + FN)
nb_precision <- cmat[2,2]/sum(cmat[,2]) # precision = TP / (TP + FP)
nb_f1 <- 2*(nb_recall*nb_precision)/(nb_recall + nb_precision)

# print
cat(
  "Baseline Accuracy: ", baseline_acc, "\n",
  "Accuracy:",  nb_acc, "\n",
  "Recall:",  nb_recall, "\n",
  "Precision:",  nb_precision, "\n",
  "F1-score:", nb_f1
)

# w smoothing ----------------

# train model on the training set using Laplace smoothing
##Recall what this does --- want to avoid having any zeroes, which happens if there's a novel word in the test set

nb_model_sm <- textmodel_nb(train_dfm, train_set$class, smooth = 1, prior = "uniform")

# evaluate on test set
predicted_class_sm <- predict(nb_model_sm, newdata = test_dfm)

# get confusion matrix
cmat_sm <- table(test_set$class, predicted_class_sm)
nb_acc_sm <- sum(diag(cmat_sm))/sum(cmat_sm) # accuracy = (TP + TN) / (TP + FP + TN + FN)
nb_recall_sm <- cmat_sm[2,2]/sum(cmat_sm[2,]) # recall = TP / (TP + FN)
nb_precision_sm <- cmat_sm[2,2]/sum(cmat_sm[,2]) # precision = TP / (TP + FP)
nb_f1_sm <- 2*(nb_recall_sm*nb_precision_sm)/(nb_recall_sm + nb_precision_sm)

# print
cat(
  "Baseline Accuracy: ", baseline_acc, "\n",
  "Accuracy:",  nb_acc_sm, "\n",
  "Recall:",  nb_recall_sm, "\n",
  "Precision:",  nb_precision_sm, "\n",
  "F1-score:", nb_f1_sm
)

# take a look at the most discriminant features (get some face validity)
posterior <- data.frame(feature = rownames(t(nb_model_sm$PcGw)), 
                    post_CRIME = t(nb_model_sm$PcGw)[,1],
                    post_SPORTS = t(nb_model_sm$PcGw)[,2])

##

head(arrange(posterior, -post_SPORTS))
head(arrange(posterior, -post_CRIME))


# what does smoothing do? More generally, reduces the "weight" place on new information (the likelihood) vis-a-vis the prior. 
plot(nb_model$PwGc[1,], nb_model_sm$PwGc[1,], xlim = c(0,0.02), ylim = c(0,0.02), xlab="No Smooth", ylab="Smooth") + abline(a = 0, b = 1, col = "red")
##don't worry about that code
#----------------------------------------
# 2 Classification using Word Scores     ---
#----------------------------------------
# Read in conservative and labour manifestos
filenames <- list.files(path = "cons_labour_manifestos")

# Party name and year are in the filename -- we can use substr to extract these to use as our docvars

party <- substr(filenames, 1, 3)
year <- substr(filenames, 4, 7)

# This is how you would make a corpus with docvars from this data
cons_labour_manifestos <- corpus(readtext("cons_labour_manifestos/*.txt"))
docvars(cons_labour_manifestos, field = c("party", "year") ) <- data.frame(cbind(party, year))

# But we're going to use a dataframe
cons_labour_df <- data.frame(text = texts(cons_labour_manifestos),
                         party = party,
                         year = as.integer(year))
colnames(cons_labour_df)

# keep vars of interest --- same as last time, but using the %>$ operator
cons_labour_df <- cons_labour_df %>% select(text, party) %>% setNames(c("text", "class"))


# what's the class distribution?
prop.table(table(cons_labour_df$class))

# randomly sample a test speech
set.seed(1984L)
ids <- 1:nrow(cons_labour_df)
ids_test <- sample(ids, 1, replace = FALSE)
ids_train <- ids[-ids_test]
train_set <- cons_labour_df[ids_train,]
test_set <- cons_labour_df[ids_test,]

# create DFMs
train_dfm <- dfm(train_set$text, remove_punct = TRUE, remove = stopwords("english"))
test_dfm <- dfm(test_set$text, remove_punct = TRUE, remove = stopwords("english"))

# Word Score model w/o smoothing ----------------
ws_base <- textmodel_wordscores(train_dfm, 
                                y = (2 * as.numeric(train_set$class == "Lab")) - 1 # Y variable must be coded on a binary x in {-1,1} scale, so -1 = Conservative and 1 = Labour
)

# Look at strongest features
lab_features <- sort(ws_base$wordscores, decreasing = TRUE)  # for labor
lab_features[1:10]

con_features <- sort(ws_base$wordscores, decreasing = FALSE)  # for conservative
con_features[1:10]

# Can also check the score for specific features
ws_base$wordscores[c("drugs", "minorities", "unemployment")]

# predict that last speech
test_set$class
predict(ws_base, newdata = test_dfm,
        rescaling = "none", level = 0.95) 

# Word Score model w smoothing ----------------
?textmodel_wordscores
ws_sm <- textmodel_wordscores(train_dfm, 
                              y = (2 * as.numeric(train_set$class == "Lab")) - 1, # Y variable must be coded on a binary x in {-1,1} scale, so -1 = Conservative and 1 = Labour
                              smooth = 1
)

# Look at strongest features
lab_features_sm <- sort(ws_sm$wordscores, decreasing = TRUE)  # for labor
lab_features_sm[1:10]

con_features_sm <- sort(ws_sm$wordscores, decreasing = FALSE)  # for conservative
con_features_sm[1:10]

# predict that last speech
test_set$class
predict(ws_base, newdata = test_dfm,
        rescaling = "none", level = 0.95) 

# Smoothing  
plot(ws_base$wordscores, ws_sm$wordscores, xlim=c(-1, 1), ylim=c(-1, 1),
     xlab="No Smooth", ylab="Smooth")


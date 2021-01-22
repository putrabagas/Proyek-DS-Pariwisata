#library yang terdapat sebuah algoritma Naive Bayes Classifier
library(e1071)
library(caret)
library(syuzhet)

#library untuk penggunaan corpus dalam cleaning data
library(tm)
library(RTextTools)
library(dplyr)

library(wordcloud)
library(shiny)
library(ggplot2)
library(plotly)


#Membersihkan dataset

#men-set working directory disuatu folder
setwd("D:/FILE KULIAH UPN/Semester 5/Prak DS/Projek/Sentiment Analysis/Sentiment Analysis")
dataReview <- readLines('universalStudio.csv')  #mengimpor data set

textid <- Corpus(VectorSource(dataReview))

removeNL <- function(y) gsub("\n", " ", y) #menghilangkan baris baru
reviewclean <- tm_map(textid, removeNL)

removeuang <- function(y) gsub("$", " ", y)  #menghilangkan simbol $
reviewclean <- tm_map(reviewclean, removeuang)

remove.all <- function(xy) gsub("[^[:alpha:][:space:]]*", "", xy)
reviewclean <- tm_map(reviewclean,remove.all)  #menghapus alpabet dan spasi

reviewclean <- tm_map(reviewclean, tolower) #huruf kapital ke huruf kecil
myStopwords = readLines("stopword_en.txt")
reviewclean <- tm_map(reviewclean,removeWords,myStopwords) #menghilangkan kata umum yang tidak memiliki makna

dataframe<-data.frame(text=unlist(sapply(reviewclean, `[`)), stringsAsFactors=F)
#View(dataframe)
write.csv(dataframe,file = 'reviewClean.csv')  #menyimpan dataset yang sudah dibersihkan


if (!require("pacman")) install.packages("pacman")
pacman::p_load(wordcloud, tm, tidyr, tidytext, syuzhet, ngram, NLP, RColorBrewer, RTextTools, e1071, caret, knitr)


#Membuat Barplot

#digunakan untuk membaca file csv yang sudah di cleaning data 
review_dataset<-read.csv("reviewClean.csv",stringsAsFactors = FALSE)

#digunakan untuk mengeset variabel cloumn text menjadi char
review <-as.character(review_dataset$text)

#memanggil NRC sentiment dictionary untuk menghitung keberadaan delapan emosi berbeda.
s<-get_nrc_sentiment(review)

review_combine<-cbind(review_dataset$text,s)

hasil_analisis<-data.frame(review_combine, stringsAsFactors=FALSE)
#View(hasil_analisis)p
write.csv(hasil_analisis,file = 'hasil_sentimen.csv')

par(mar=rep(3,4))
a<- barplot(colSums(s),col=rainbow(10),ylab='count',main='sentiment analisis')
iki_ba <- a


df<-read.csv("reviewClean.csv",stringsAsFactors = FALSE)
glimpse(df)

set.seed(20)
df<-df[sample(nrow(df)),]
df<-df[sample(nrow(df)),]
glimpse(df)

corpus<-Corpus(VectorSource(df$text))

#fungsinya untuk membersihkan data data yang tidak dibutuhkan 
corpus.clean<-corpus%>%
  tm_map(content_transformer(tolower))%>%
  tm_map(removePunctuation)%>%
  tm_map(removeNumbers)%>%
  tm_map(removeWords,stopwords(kind="en"))%>%
  tm_map(stripWhitespace)
dtm<-DocumentTermMatrix(corpus.clean)

dtm.train<-dtm[1:75,]
dtm.test<-dtm[76:104,]

corpus.clean.train<-corpus.clean[1:75]
corpus.clean.test<-corpus.clean[76:104]

dim(dtm.train)
fivefreq<-findFreqTerms(dtm.train,5)
length(fivefreq)

dtm.train.nb<-DocumentTermMatrix(corpus.clean.train,control = list(dictionary=fivefreq))

#dim(dtm.train.nb)

dtm.test.nb<-DocumentTermMatrix(corpus.clean.test,control = list(dictionary=fivefreq))

dim(dtm.test.nb)

convert_count <- function(x){
  y<-ifelse(x>0,1,0)
  y<-factor(y,levels=c(0,1),labels=c("no","yes"))
  y
}
trainNB<-apply(dtm.train.nb,2,convert_count)
testNB<-apply(dtm.test.nb,1,convert_count)

wordcloud(corpus.clean,min.freq = 4,max.words=30,random.order=FALSE,colors=brewer.pal(8,"Dark2"))


#Shiny

review2<- read.csv("reviewClean.csv", header = TRUE)
reviewid <- review2$text

analisis <- read.csv("hasil_sentimen.csv", header = TRUE)  #membuka data hasil sentimen

review_dataset<-read.csv("reviewClean.csv",stringsAsFactors = FALSE)

ui <- fluidPage(
  titlePanel("Sentiment Analysis Universal Studios Florida Reviews"),
  mainPanel(
    tabsetPanel(type = "tabs",
                tabPanel("Data Review", DT::dataTableOutput('tbl')), # Output Data Dalam Tabel
                tabPanel("Data Analisis Sentimen", DT::dataTableOutput('sentiment')),
                tabPanel("Plot", plotOutput("plot")), 
                
                tabPanel("Wordcloud", plotOutput("Wordcloud"))
    )
  )
)
# SERVER
server <- function(input, output) {
  
  # Output Data
  output$tbl = DT::renderDataTable({
    DT::datatable(review2, options = list(lengthChange = FALSE))
  })
  
  # Output Data Sentiment
  output$sentiment = DT::renderDataTable({
    DT::datatable(analisis, options = list(lengthChange = FALSE))
  })
  
  # Output Plot
  output$plot <- renderPlot({review_dataset<-read.csv("reviewClean.csv",stringsAsFactors = FALSE)
  
  review<-as.character(review_dataset$text)
  
  s<-get_nrc_sentiment(review)
  
  review_combine<-cbind(review_dataset$text,s)
  par(mar=rep(3,4))
  barplot(colSums(s),col=rainbow(10),ylab='count',main='sentiment analisis')
  }, height=400)
  
  # Output Wordcloud
  output$Wordcloud <- renderPlot({
    set.seed(20)
    df<-df[sample(nrow(df)),]
    df<-df[sample(nrow(df)),]
    glimpse(df)
    
    
    #df$index=as.factor(df$index)
    corpus<-Corpus(VectorSource(df$text))
    
    #fungsinya untuk membersihkan data data yang tidak dibutuhkan 
    corpus.clean<-corpus%>%
      tm_map(content_transformer(tolower))%>%
      tm_map(removePunctuation)%>%
      tm_map(removeNumbers)%>%
      tm_map(removeWords,stopwords(kind="en"))%>%
      tm_map(stripWhitespace)
    dtm<-DocumentTermMatrix(corpus.clean)
    
    df.train<-df[1:75,]
    df.test<-df[76:104,]
    
    dtm.train<-dtm[1:75,]
    dtm.test<-dtm[76:104,]
    
    corpus.clean.train<-corpus.clean[1:75]
    corpus.clean.test<-corpus.clean[76:104]
    
    dim(dtm.train)
    fivefreq<-findFreqTerms(dtm.train,5)
    length(fivefreq)
    
    dtm.train.nb<-DocumentTermMatrix(corpus.clean.train,control = list(dictionary=fivefreq))
    
    #dim(dtm.train.nb)
    
    dtm.test.nb<-DocumentTermMatrix(corpus.clean.test,control = list(dictionary=fivefreq))
    
    dim(dtm.test.nb)
    
    convert_count <- function(x){
      y<-ifelse(x>0,1,0)
      y<-factor(y,levels=c(0,1),labels=c("no","yes"))
      y
    }
    trainNB<-apply(dtm.train.nb,2,convert_count)
    testNB<-apply(dtm.test.nb,1,convert_count)
    
    classifier<-naiveBayes(trainNB,df.train$text,laplace = 1)
    wordcloud(corpus.clean,min.freq = 4,max.words=30,random.order=F,colors=brewer.pal(8,"Dark2"))
  })
}
shinyApp(ui = ui, server = server)



# Import data
data <- read.csv(file="~/plots/sync.csv", header=TRUE,
                 check.names = FALSE)

# Change data to wide format
library(reshape2)
data <- melt(data)

# Assign x and y variables to a data frame
x <- data$variable
y <- data$value
z <- data$Diagnosis

# Plot data
boxplot(y~z*x, las=2, axes=FALSE, at=c(1.1, 1.9, 3.1, 3.9),
        main="Cell cycle synchrony",
        xlab="Cell division stages", ylab="Time between divisions (hours)", 
        font.lab = 2, col=c("grey", "white"), outpch=1, outcex=.7)
box()
axis(2, at=seq(0, 20, 5), las=1)
axis(1, at=c(1.5, 3.5), labels=c("3-4 cell", "5-6 cell"))

legend(x=3.29, y=19.7, legend=c("Aneuploid", "Euploid"), 
       fill=c("grey", "white"))

# Import data
data <- read.csv(file="~/plots/boxplot.csv", header=TRUE,
                 check.names = FALSE)

# Change data to wide format
library(reshape2)
data <- melt(data, timevar=c("Euploid", "Aneuploid"))
data

# Assign x and y variables to a data frame
x <- data$variable
y <- data$value
z <- data$Diagnosis

# Plot data
boxplot(y~z + x, las=2, at=c(1.1, 1.9, 3.1, 3.9, 5.1, 5.9, 7.1, 7.9, 9.1, 9.9, 11.1, 11.9, 13.1, 13.9),
        axes=FALSE, main="Chronology of developmental events",
        xlab="Embryonic event", ylab="Timing of event (hours after insemination)", font.lab = 2,
        col=c("grey", "white"), outpch=1, outcex=.5, ylim=c(20, 140))
box()
axis(2, at=seq(0, 200, 20), las=1)
axis(1, at=seq(1.5, 13.5, 2), labels=c("2 cell", "3 cell",
                                       "4 cell", "5 cell",
                                       "6 cell", "tSB",
                                       "tB"))

legend(x=.7, y=136, legend=c("Aneuploid", "Euploid"), 
       fill=c("grey", "white"))


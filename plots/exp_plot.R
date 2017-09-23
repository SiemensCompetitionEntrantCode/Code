data <- read.csv(file="~/plots/expansion.csv", header=TRUE,
                 check.names = FALSE)
library(reshape2)
data <- melt(data, id.var = "Diagnosis")

x <- data$variable
y <- data$value
z <- data$Diagnosis

plot(1, 1, type='n', las=1, ylim=c(12, 20), xlim=c(0.9, 6.1),
     axes=F, xlab="Time from blastocyst formation (hours)",
     ylab="Blastocyst cross-sectional area (thousand μm²)",
     main="Blastocoel cavity expansion")

lines(lowess(x[z=="Aneuploid"], y[z=="Aneuploid"]), 
      lwd = 2, lty = 2)
lines(lowess(x[z=="Euploid"], y[z=="Euploid"]), 
      lwd = 2)

# ----------points-----------
data <- read.csv(file="~/plots/expansion_points.csv", header=TRUE,
                 check.names = FALSE)

x <- data$time
y <- data$avg
z <- data$diag
e <- data$e

points(x[z=="Euploid"], y[z=="Euploid"], cex = 1.1, pch = 19)
points(x[z=="Aneuploid"], y[z=="Aneuploid"], cex=1.1, lwd = 1.5)

segments(x, y-e,x, y+e, lwd = 1.3)
width = 0.05

segments(x-width, y-e, x+width, y-e, lwd = 1.3)
segments(x-width, y+e, x+width, y+e, lwd = 1.3)

box()
axis(2, at=seq(0, 20, 1), las=1)
axis(1, at=seq(1, 6, 1), labels=c("0", "2", "4", "6", "8", "10"))

legend(x=.98, y=19.45, legend=c("Euploid", "Aneuploid"), 
       lty=c(1,2), lwd=3)


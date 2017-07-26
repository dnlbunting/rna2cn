library(Rsamtools)
library(Biostrings)
library(DNAcopy)

getRefGenome <- function(fasta=FASTA)
{
    dna <- readDNAStringSet(fasta,format="fasta")
    names(dna) <- lapply(names(dna), function(x) strsplit(x, ' ')[[1]][[1]])
    dna<-lapply(c(1:22,"X","Y","MT"), function(x) dna[[as.character(x)]])
    return(dna)
}

getStartsEndsBED <- function(path, chr)
{
    t <- read.table(path,header=F)
    colnames(t) <- c('CHR', 'START', 'END', 'NAME')
    subt <- t[t$CHR==chr,]
    starts <- as.numeric(as.character(subt$START))
    ends <- as.numeric(as.character(subt$END))
    list(starts=starts,ends=ends)
}

gcTrack <- function(chr,
                    starts,
                    ends)
{
    gc <- rowSums(letterFrequencyInSlidingView(dna[[chr]],
                                               5000,
                                               c("G","C")))/5000
    if(ends[length(ends)]>length(gc))
    {
        ends[length(ends)] <- length(gc)
    }
    gc <- sapply(1:length(starts),function(x) mean(gc[starts[x]:ends[x]], na.rm=TRUE))
    names(gc) <- paste("bin",1:length(starts),sep="-")
    gc
}

getCoverageTrack <- function(bamPath, chr, starts,ends, sampleDepth)
{
    sbp <- ScanBamParam(flag=scanBamFlag(isDuplicate=FALSE),
                        which=GRanges(paste0(CHRSTRING,chr),
                                      IRanges(starts,ends)))
    coverageTrack <- countBam(bamPath,param=sbp)
    coverageTrack$records <- (coverageTrack$records+1)/sampleDepth #mean(coverageTrack$records)
    return(coverageTrack)
}

smoothCoverageTrackAll <- function(lCT,lSe,lGCT)
{
    allRec <- unlist(lapply(lCT,function(x) log10(x$records)))
    allGC <- unlist(lGCT)
    starts <- c(0,cumsum(sapply(lCT,nrow)[-c(length(lCT))]))+1
    ends <- cumsum(sapply(lCT,nrow))
    smoothT <- loess(allRec~allGC)
    for(i in 1:length(lCT))
    {
        lCT[[i]] <- cbind(lCT[[i]],smoothT$fitted[starts[i]:ends[i]],
                          smoothT$residuals[starts[i]:ends[i]])
        colnames(lCT[[i]])[(ncol(lCT[[i]])-1):ncol(lCT[[i]])] <- c("fitted","smoothed")
    }
    return(lCT)
}

segmentTrack <- function(covtrack,
                         chr,
                         starts,
                         ends=NA,
                         sd=0,
                         min.width=5)
{
    ## maploc <- seq(1,pas*(length(covtrack)),pas)
    ## maploc <- maploc[]
    covtrack <- covtrack*rnorm(length(covtrack),mean=1,sd=sd)
    cna <- CNA(covtrack,chr=rep(chr,length(covtrack)),
               maploc=starts,data.type="logratio")
    cna.smoothed <- smooth.CNA(cna)
    segment(cna.smoothed,min.width=min.width)
}

segment <- function (x, weights = NULL,
                     alpha = 0.01,
                     nperm = 10000,
                     p.method = c("hybrid",
                                  "perm"),
                     min.width = 2,
                     kmax = 25,
                     nmin = 200,
                     eta = 0.05,
                     sbdry = NULL,
                     trim = 0.025,
                     undo.splits = c("none", "prune",
                                     "sdundo"),
                     undo.prune = 0.05,
                     undo.SD = 3,
                     verbose = 1)
{
    require(DNAcopy)
    if (!inherits(x, "CNA"))
        stop("First arg must be a copy number array object")
    call <- match.call()
    ##    if (min.width < 2 | min.width > 5)
    ##        stop("minimum segment width should be between 2 and 5")
    if (nmin < 4 * kmax)
        stop("nmin should be >= 4*kmax")
    if (missing(sbdry)) {
        if (nperm == 10000 & alpha == 0.01 & eta == 0.05) {
            if (!exists("default.DNAcopy.bdry"))
                data(default.DNAcopy.bdry, package = "DNAcopy",
                     envir = environment())
            sbdry <- default.DNAcopy.bdry
        }
        else {
            max.ones <- floor(nperm * alpha) + 1
            sbdry <- getbdry(eta, nperm, max.ones)
        }
    }
    weighted <- ifelse(missing(weights), FALSE, TRUE)
    if (weighted) {
        if (length(weights) != nrow(x))
            stop("length of weights should be the same as the number of probes")
        if (min(weights) <= 0)
            stop("all weights should be positive")
    }
    sbn <- length(sbdry)
    nsample <- ncol(x) - 2
    sampleid <- colnames(x)[-(1:2)]
    uchrom <- unique(x$chrom)
    data.type <- attr(x, "data.type")
    p.method <- match.arg(p.method)
    undo.splits <- match.arg(undo.splits)
    segres <- list()
    segres$data <- x
    allsegs <- list()
    allsegs$ID <- NULL
    allsegs$chrom <- NULL
    allsegs$loc.start <- NULL
    allsegs$loc.end <- NULL
    allsegs$num.mark <- NULL
    allsegs$seg.mean <- NULL
    segRows <- list()
    segRows$startRow <- NULL
    segRows$endRow <- NULL
    for (isamp in 1:nsample) {
        if (verbose >= 1)
            cat(paste("Analyzing:", sampleid[isamp], "\n"))
        genomdati <- x[, isamp + 2]
        ina <- which(is.finite(genomdati))
        genomdati <- genomdati[ina]
        trimmed.SD <- sqrt(DNAcopy:::trimmed.variance(genomdati, trim))
        chromi <- x$chrom[ina]
        if (weighted) {
            wghts <- weights[ina]
        }
        else {
            wghts <- NULL
        }
        sample.lsegs <- NULL
        sample.segmeans <- NULL
        for (ic in uchrom) {
            if (verbose >= 2)
                cat(paste("  current chromosome:", ic, "\n"))
            segci <- DNAcopy:::changepoints(genomdati[chromi == ic], data.type,
                                            alpha, wghts, sbdry, sbn, nperm, p.method, min.width,
                                            kmax, nmin, trimmed.SD, undo.splits, undo.prune,
                                            undo.SD, verbose)
            sample.lsegs <- c(sample.lsegs, segci$lseg)
            sample.segmeans <- c(sample.segmeans, segci$segmeans)
        }
        sample.nseg <- length(sample.lsegs)
        sample.segs.start <- ina[cumsum(c(1, sample.lsegs[-sample.nseg]))]
        sample.segs.end <- ina[cumsum(sample.lsegs)]
        allsegs$ID <- c(allsegs$ID, rep(isamp, sample.nseg))
        allsegs$chrom <- c(allsegs$chrom, x$chrom[sample.segs.end])
        allsegs$loc.start <- c(allsegs$loc.start, x$maploc[sample.segs.start])
        allsegs$loc.end <- c(allsegs$loc.end, x$maploc[sample.segs.end])
        allsegs$num.mark <- c(allsegs$num.mark, sample.lsegs)
        allsegs$seg.mean <- c(allsegs$seg.mean, sample.segmeans)
        segRows$startRow <- c(segRows$startRow, sample.segs.start)
        segRows$endRow <- c(segRows$endRow, sample.segs.end)
    }
    allsegs$ID <- sampleid[allsegs$ID]
    allsegs$seg.mean <- round(allsegs$seg.mean, 4)
    allsegs <- as.data.frame(allsegs)
    allsegs$ID <- as.character(allsegs$ID)
    segres$output <- allsegs
    segres$segRows <- as.data.frame(segRows)
    segres$call <- call
    if (weighted)
        segres$weights <- weights
    class(segres) <- "DNAcopy"
    segres
}

getTrackForAll <- function(bamfile,
                           window,
                           lSe=NULL,
                           lGCT=NULL,
                           lCT=NULL,
                           lCTS=NULL,
                           allchr=1:22,
                           sdNormalise=0)
{
    
    if(is.null(lSe)) {
        print("get Start-End of segments")
        lSe <- lapply(allchr,function(chr) getStartsEndsBED(window, chr))
    }
                          
    if(is.null(lCT) && is.null(lCTS)){ 
        print("get Coverage Track")
        lCT <- lapply(allchr, function(chr) getCoverageTrack(bamPath=bamfile,
                                                         chr=paste0(chr),
                                                         lSe[[chr]]$starts,
                                                         lSe[[chr]]$ends))
    }
                      
    if (is.null(lCTS)){
        if(is.null(lGCT)) {
            print("get GC content")
            lGCT <- lapply(allchr,function(chr) gcTrack(chr,lSe[[chr]]$starts,lSe[[chr]]$ends))
        }

        print("correct for GC content")
        lCTS <- smoothCoverageTrackAll(lCT,lSe,lGCT)
    }

    print("segment Tracks")
    lSegs <- lapply(1:length(lCTS),function(x)
    {
        require(DNAcopy)
        segments<- segmentTrack(lCTS[[x]]$normalised,
                                chr=paste0(x),
                                sd=sdNormalise,
                                lSe[[x]]$starts,
                                lSe[[x]]$ends)
    })
    names(lSegs) <- paste0(1:length(lCT))
    tracks <- list(lCTS=lCTS,lSegs=lSegs)

    return(tracks)
}

plotGCCorrection <- function(lCT,lGCT)
{
                   
    allRec <- unlist(lapply(lCT,function(x) log10(x$records)))
    allGC <- unlist(lGCT)

    smoothT <- loess(allRec~allGC)

    plot(allGC,allRec,pch=19,cex=.4,col=rgb(0,0,0,.3))
    ord <- order(smoothT$x,decreasing=F)
    points(smoothT$x[ord],smoothT$fitted[ord],type="l",col=rgb(.5,0,0,.5),lwd=2)
                            
}
                            
plotAllGenome <- function(tracks,
                          colSeg=rgb(.5,.2,.5,.7),
                          lwdSeg=2,
                          ...)
{
    breaks <- c(0,cumsum(sapply(tracks$lSegs,function(x) max(x$output$loc.end))))/1000000
    plot(0,0,col=rgb(0,0,0,0),
         xaxt="n",
         yaxt="n",
         xlim=c(0,max(breaks)),
         ylim=c(0,5),
         xlab="Genomic Position",
         ylab="Copies",
         frame=F,...)
    axis(side=1)
    axis(side=2)
    for(i in 1:length(tracks$lSegs))
    {
        segments(tracks$lCTS[[i]]$start/1000000+breaks[i],
                 2*10**tracks$lCTS[[i]]$normalised,
                 tracks$lCTS[[i]]$end/1000000+breaks[i],
                 2*10**tracks$lCTS[[i]]$normalised,col=rgb(.7,.7,.7,.6))
        segments(tracks$lSegs[[i]]$output$loc.start/1000000+breaks[i],
                 round(2*10**tracks$lSegs[[i]]$output$seg.mean),
                 tracks$lSegs[[i]]$output$loc.end/1000000+breaks[i],
                 round(2*10**tracks$lSegs[[i]]$output$seg.mean),
                 lwd=lwdSeg,
                 col=colSeg)
    }
    abline(h=0,v=breaks,lwd=1,lty=2,col=rgb(.6,.6,.6,.4))
    text(x=breaks[2:length(breaks)]-25,y=1,names(breaks)[2:length(breaks)],cex=.4)
}
                                
SegsToBed <- function(track, bamfile){
    fout <- file(paste0(dirname(bamfile), '/', strsplit(basename(bamfile), '_')[[1]][[1]], "_CN.bed"), open='w')
    for (chr in ALLCHR){
        df <- track$lSegs[[chr]][[2]]
        for (i in 1:nrow(df)){
            row <- df[i,]
            write(paste0(chr, "\t", row$loc.start, "\t", row$loc.end, "\t", round(2*10**(row$seg.mean))), file=fout)
        }
    }
    close(fout)
}
                                
indexBams <- function(bams,mc.cores=10)
{
    require(parallel)
    mclapply(bams,function(x) {
        cmd <- paste0("module load SAMtools; samtools index ", x)
        system(cmd,wait=T)
    },mc.cores=mc.cores)
}
APA analysis refers to the examination of Alternative Polyadenylation (APA), which is a post-transcriptional regulatory mechanism of gene expression. In APA, mRNA molecules are polyadenylated at different sites at the 3' end, resulting in mRNA variants with different lengths of the 3' untranslated region (3' UTR). This selective polyadenylation can influence the stability, translational efficiency, and cellular localization of the mRNA, thereby finely tuning gene expression.

we can use DaPars to peform APA analysis

github：[ZhengXia/dapars: DaPars(Dynamic analysis of Alternative PolyAdenylation from RNA-seq) (github.com)](https://github.com/ZhengXia/dapars?tab=readme-ov-file)
There is also DAPARS2 now, here we only take DaPars 1.0 as an example
workflow：

![work_flow](/img/APA_workflow.png)

# installation:

```jsx
##Prerequisite: Bedtools; python3; numpy; scipy
install DaPars:
tar zxf DaPars-VERSION.tar.gz
```

# step1: get your input file

## 1.1 get reference gene annotation from UCSC TABLE BROWSER([Table Browser (ucsc.edu)](https://genome.ucsc.edu/cgi-bin/hgTables))

Gene 3'UTR regions can be extracted from reference gene annotation, any reference gene annotations available at UCSC table browser are supported by DaPars2. Follow the guide shown in the figure below to get the gene annotation and the ID mapping between RefSeq gene id and gene symbol.

for example:

![Untitled](/img/APA1.png)

![Untitled](/img/APA2.png)

## 1.2 BedGraph files store the reads alignment result. it can be generated from BAM file from RNA-seq alignment tool such as TopHat. One way is to use [BedTools](https://github.com/arq5x/bedtools2) with following command:

ps: this step need chrome length of your species

my codes:

```jsx
genomeCoverageBed -bg -ibam HKCI10-sgA5-3_1.sorted.bam -g ../mm10_m25_gencode.len -split > HKCI10-sgA5-3_1.bedgraph
```

# step 2:

Generate region annotation
DaPars will use the extracted distal polyadenylation sites to infer the proximal polyadenylation sites based on the alignment wiggle files of two samples. The output in this step will be used by the next step.

my codes:

```jsx
python /DaPars_Extract_Anno.py -b mm10_m25.bed -s mm10_m25_Refseq_id.txt -o mm10_m25_extracted_3UTR.bed
```

# step 3:

**main function to get final result**

Run this function to get the final result. The configure file is the only parameter for DaPars_main.py, which stores all the parameters.

this step need a config ,then run the codes

```jsx
#my config
Annotated_3UTR=mm10_m25_extracted_3UTR.bed
Group1_Tophat_aligned_Wig=Ctrl_1.bedgraph,Ctrl_2.bedgraph,Ctrl_3.bedgraph
Group2_Tophat_aligned_Wig=LV_1.bedgraph,LV_2.bedgraph,LV_3.bedgraph
Output_directory=DaPars/
Output_result_file=DaPars

#Parameters
Num_least_in_group1=3
Num_least_in_group2=3
Coverage_cutoff=30
FDR_cutoff=0.05
PDUI_cutoff=0.5
Fold_change_cutoff=0.59

#my codes:
python DaPars_main.py config.txt
```

# output format:

![Untitled](/img/APAoutput.png)

ps: Group1 of config is the group A of output

# Visualization：
Here i select differentially expressed genes based on Pvalue ≤ 0.1, Pvalue ≤ 0.05 is also worked
my code:
```jsx
PDUIPVal <- read.table("DaPars_All_Prediction_Results.txt",row.names=1,header=T)
head(PDUIPVal)
print(colnames(PDUIPVal))

##select differentially expressed genes based on Pvalue ≤ 0.1
PDUIPVal[which(abs(PDUIPVal[,24]) <=0.1),'DEG0.1'] <- 'no diff'
PDUIPVal[which((abs(PDUIPVal[,24])>=0.1)&(PDUIPVal[,22]>PDUIPVal[,23])),'DEG0.1'] <-
  'Shortened(0.1)'
PDUIPVal[which((abs(PDUIPVal[,24])>=0.1)&(PDUIPVal[,22]<PDUIPVal[,23])),'DEG0.1'] <-
  'Lengthened(0.1)'
table(PDUIPVal$DEG0.1)
pdf("fig1-pduipaad.pdf",width=6.5,height=5.7)
par(cex.lab=1.3,cex.axis=1.5)
par(mar=c(5,5,3,3))
col0<-rep("#00000022",nrow(PDUIPVal))
##colors is red and blue
col0[(abs(PDUIPVal[,24])>= 0.1)&(PDUIPVal[,22]>PDUIPVal[,23])] <- "#FF000090"
col0[(abs(PDUIPVal[,24])>= 0.1)&(PDUIPVal[,22]<PDUIPVal[,23])] <- "#0000FF90"
pch0<-rep(21,nrow(PDUIPVal))
par(pty="s")
plot(PDUIPVal[,22:23],bg=col0,pch=pch0,col=NA,cex=1,axes=FALSE,xlab="",ylab="")
title(xlab="Ctrl_Mean_PDUI",ylab="LV_Mean_PDUI",line= 2.3,cex.lab= 1)
title(main="",line= 0.7)
box();axis(1,at=c(0,.5,1));axis(2,at=c(0,.5,1))
legend("topleft", c("Lengthened", "Shortened"), pt.bg=c("#0000FF90","#FF000090"),
       pch=c(21,21), cex=1)
abline(c(.1,1),lty=2,lwd=2)
abline(c(-.1,1),lty=2,lwd=2)
par(pty="m")
plot((PDUIPVal[,24]),-log10(PDUIPVal[,26]),xlim=c(-.6,.6),ylim=c(0,20),
     xlab="",ylab="",col=NA,pch=pch0,bg=col0,cex=1)
title(xlab="Change in PDUI (Lv-Ctrl)",ylab="p-value (BH, -log10)",line= 2.3,cex.lab= 1.3)
abline(v=0.1*c(-1,1),lty=2,lwd=2)
dev.off()
```
here is my plot

![Untitled](/img/APApng-1.png)
![Untitled](/img/APApng-2.png)
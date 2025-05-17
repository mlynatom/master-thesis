# Define an array of tasks

# The task's llh score is averaged by default
export TASKS=(
    # "benczechmark_agree"                   #0
    # "benczechmark_belebele"                #1
    # "benczechmark_czechnews"               #2
    # "benczechmark_snli" #3
    # "benczechmark_subjectivity"            #4
    # "benczechmark_propaganda_argumentace"  #5
    # "benczechmark_propaganda_fabulace"     #6
    # "benczechmark_propaganda_nazor"        #7
    # "benczechmark_propaganda_strach" #8
    # "benczechmark_propaganda_zamereni"     #9
    # "benczechmark_propaganda_demonizace"   #10
    # "benczechmark_propaganda_lokace"       #11
    # "benczechmark_propaganda_relativizace" #12
    # "benczechmark_propaganda_vina"         #13
    # "benczechmark_propaganda_zanr"         #14
    # "benczechmark_propaganda_emoce"        #15
    "benczechmark_propaganda_nalepkovani"  #16
    "benczechmark_propaganda_rusko"        #17
    # "benczechmark_sentiment_mall"          #18
    # "benczechmark_sentiment_fb"            #19
    # "benczechmark_sentiment_csfd"          #20
    # "benczechmark_summarization" #21
    # "benczechmark_grammarerrorcorrection"  #22
    # "benczechmark_cs_naturalquestions" #23
    # "benczechmark_cs_sqad32"           #24
    # "benczechmark_cs_triviaQA"         #25
    # "benczechmark_csfever_nli"             #26
    # "benczechmark_ctkfacts_nli"            #27
    # "benczechmark_cs_ner" #28
    # "benczechmark_hellaswag"               #29
    # "benczechmark_klokan_qa"              #30
    # "benczechmark_cs_court_decisions_ner" #31
    # "benczechmark_umimeto_qa"              #32
    # "benczechmark_cermat_mc"               #33
    # "benczechmark_cermat_qa" #34
    # "benczechmark_history_ir"              #35
    # "benczechmark_histcorpus"              #36
    # "benczechmark_essay"                   #37
    # "benczechmark_fiction"                 #38
    # "benczechmark_capek"                   #39
    # "benczechmark_correspondence"          #40
    # "benczechmark_havlicek"                #41
    # "benczechmark_speeches"                #42
    # "benczechmark_spoken"                  #43
    # "benczechmark_dialect"                 #44
)

# Define tasks that require summing of logprobs
SUM_LOGPROBS=(
  "benczechmark_histcorpus"
  "benczechmark_essay"
  "benczechmark_fiction"
  "benczechmark_capek"
  "benczechmark_correspondence"
  "benczechmark_havlicek"
  "benczechmark_speeches"
  "benczechmark_spoken"
  "benczechmark_dialect"
)
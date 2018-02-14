
# Experiments for DBS on "bi-grams" as concepts
python pipeline.py --oracle_type=accept_reject --data_set=DBS --language=german --summarizer_type=feedback --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=ilp_feedback --data_set=DBS --language=german --summarizer_type=feedback --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning --data_set=DBS --language=german --summarizer_type=feedback --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning2 --data_set=DBS --language=german --summarizer_type=feedback --rouge=rouge/RELEASE-1.5.5/ 

# Experiments for DBS on "phrase" as concepts
python pipeline.py --oracle_type=accept_reject --data_set=DBS --language=german --summarizer_type=feedback --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=ilp_feedback --data_set=DBS --language=german --summarizer_type=feedback --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning --data_set=DBS --language=german --summarizer_type=feedback --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning2 --data_set=DBS --language=german --summarizer_type=feedback --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 

# Experiments for DUC2001 on "bi-grams" as concept type  -> change data_set to "DUC2002", "DUC2004"
python pipeline.py --oracle_type=accept_reject --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=ilp_feedback --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning2 --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --rouge=rouge/RELEASE-1.5.5/ 

# Experiments for DUC2001 on "phrases" as concept type -> change data_set to "DUC2002", "DUC2004"
python pipeline.py --oracle_type=accept_reject --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=ilp_feedback --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 
python pipeline.py --oracle_type=active_learning2 --data_set=DUC2001 --language=english --summarizer_type=feedback --summary_size=100 --parser_type=parse --rouge=rouge/RELEASE-1.5.5/ 


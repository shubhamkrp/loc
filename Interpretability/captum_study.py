def compute_captum_attributions(
   gnn,
   data,
   feature_ranges,
   empath_cats,
   trigram_list,
   target_class=0
):
   torch.cuda.empty_cache()


   device = torch.device("cuda:1")


   gnn = deepcopy(gnn).to(device)
   gnn.eval()


   x = data.x.detach().to(device)
   edge_index = data.edge_index.to(device)


   wrapper = GNNWrapper(gnn, edge_index).to(device)
   ig = IntegratedGradients(wrapper)


   # ---- Baseline (zeros) ----
   baseline = torch.zeros_like(x)


   # ---- Compute attributions ----
   attributions = ig.attribute(
       inputs=x,
       baselines=baseline,
       target=target_class,
       internal_batch_size=32  # <<< CRITICAL for memory
   )


   attributions = attributions.abs().mean(dim=0).cpu().numpy()


   results = {}
   for name, (s, e) in feature_ranges.items():
       results[name] = float(attributions[s:e].mean())


   # ---- Top trigrams ----
   trig_attr = attributions[
       feature_ranges["trigrams"][0]:feature_ranges["trigrams"][1]
   ]
   top_trigrams = sorted(
       zip(trigram_list, trig_attr),
       key=lambda x: x[1],
       reverse=True
   )[:10]


   # ---- Top empath emotions ----
   emp_attr = attributions[
       feature_ranges["empath"][0]:feature_ranges["empath"][1]
   ]
   top_emotions = sorted(
       zip(empath_cats, emp_attr),
       key=lambda x: x[1],
       reverse=True
   )[:10]


   # ---- Cleanup ----
   del x, baseline, attributions, wrapper, gnn
   torch.cuda.empty_cache()



  ##### USAGE:
  # attr_scores, top_trigrams, top_emotions = compute_captum_attributions(
  #              gnn=gnn,
  #              data=test_data,
  #              feature_ranges=FEATURE_RANGES,
  #              empath_cats=empath_cats,
  #              trigram_list=trigram_list,
  #              target_class=0  # IP
  #          )

   return results, top_trigrams, top_emotions

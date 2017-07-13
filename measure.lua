function nn.get_bleu(cand, ref, n)
   n = n or 4
   local smooth = 1
   if type(cand) ~= 'table' then
      cand = cand:totable()
   end
   if type(ref) ~= 'table' then
      ref = ref:totable()
   end
   local res = nn.utils.get_ngram_prec(cand, ref, n)
   local brevPen = math.exp(1-math.max(1, #ref/#cand))
   local correct = 0
   local total = 0
   local bleu = 1
   for i = 1, n do
      if res[i][1] > 0 then
         if res[i][2] == 0 then
            smooth = smooth*0.5
            res[i][2] = smooth
         end
         local prec = res[i][2]/res[i][1]
         bleu = bleu * prec
      end
   end
   bleu = bleu^(1/n)
   return bleu*brevPen
end

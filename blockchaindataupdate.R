# blockchain data
rm(list=ls())

require(ggplot2)
require(zoo)
require(purrr)
require(stringr)
require(data.table)

# setwd("path-to-'cryptoarchive'")

#cutoff at june 2018
#counted number of coins from remaining sample
#num coins = max then stop

# configuration
upperbound <- 0.05 # 0.025
freq <- 30 #mins # 60
maxcutoff <- 480 #mins

datlist <- list.files("cryptoarchive")
dat <- map(datlist, ~fread(paste0("cryptoarchive/", .x))) %>%
  set_names(datlist) %>%  
  rbindlist(idcol = "coinidx", fill = T)

dat[, ':='(V2=NULL, V3=NULL, V4=NULL, V5=NULL, V6=NULL, V7=NULL)]
setnames(dat, "V1", "rowbycoin")
dat[, coinidx := str_remove_all(coinidx, ".csv")]

# brute force get datetimes
dat[, datetime := as.POSIXct(as.numeric(time) / 1000, origin="1970-01-01", tz="GMT")]
dat[, timediff := datetime - shift(datetime), by = coinidx]
dat <- dat[!(is.na(datetime) | is.na(timediff))]

dat[, samplelength := .N, by = coinidx]
dat[, n30 := sum(timediff >= freq) / samplelength, by = coinidx]

# filter data by max trading time diff

dat[, maxdiff := max(timediff, na.rm = T), by = coinidx]
dmax <- unique(dat[, .SD, .SDcols = c("coinidx", "maxdiff")])

print(dmax)
print(dmax[maxdiff == min(dmax$maxdiff)])

datsub <- dat[maxdiff <= maxcutoff]
print(uniqueN(datsub$coinidx))

# check concentration of sample sizes

summary(datsub$samplelength)
unique(datsub$coinidx)

# for each coin, find sample window, fill in number of 30min slots, spit out new dt 

fulldat <- datsub[, {
  timewin = max(datetime) - min(datetime)
  gridsize = timewin * 24 * 2 + 1
  .(
    dtime = c(1:gridsize),
    coin = coinidx,
    min = min(datetime),
    max = max(datetime),
    samplelength = unique(samplelength),
    n30 = unique(n30)
  )
},
by = coinidx]

print(min(fulldat[, .(maxdtime = max(dtime, na.rm = T)), by = coin]$maxdtime))
fulldat[, ':='(coinidx = NULL, dtime = NULL)]
finalnames <- fulldat$coin %>% unique

finaldat <- map(finalnames, ~ {
  sub <- fulldat[coin == .x]
  if (sub$n30 < upperbound) {
    tmpcol <- sapply(seq(sub$samplelength), function(n) sub$min + freq * 60 * (n - 1)) %>% unique
    cat(paste(unique(sub$coin), unique(sub$samplelength), "\n"))
    set(sub, j = "time", value = tmpcol)
    return(sub)
  }
}) %>% rbindlist
  
finalnames <- finaldat$coin %>% unique
finaldat[, datetime30 := as.POSIXct(time, origin="1970-01-01", tz="GMT")][, markercol := 1L]

finaldat[, ':='(time = NULL)]
setnames(datsub, "coinidx", "coin")

tdt <- rbind(datsub[, .(coin, datetime, close)], 
           finaldat[, .(coin, datetime=datetime30, markercol)], fill = T)[order(datetime)]

setkeyv(tdt, c("datetime", "coin")) 

# now just do na.locf on t, then throw out everything outside of the 30min markers
final <- tdt[, close := na.locf(close, na.rm = F), by = coin][markercol == 1L]# forward

final <- final[!is.na(close)][, markercol := NULL]
setnames(final, "close", "price")

final[, returns := log(price) - log(shift(price)) , by = coin]

final <- na.trim(final)

final[, newdate := ceiling_date(datetime, units="hour")]

finalsub <- final[datetime > "2018-06-01"]
finalsub[, count := .N, by = coin]
finalsub <- finalsub[count==max(count)]

saveRDS(final, file = "coindata30.RDS")

# Convert from Intervals and numbers to LazySets
to_set(a::AbstractArray) = reduce(×, to_set.(a))
to_set(i::IntervalArithmetic.Interval) = Interval(i)
to_set(s::LazySet) = s
to_set(x) = Singleton([x])

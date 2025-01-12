use super::GroupByExt;
use polars::prelude::*;

impl GroupByExt for GroupBy<'_> {
    fn keys_sliced_seq(&self, slice: Option<(i64, usize)>, by: Vec<Series>) -> Vec<Series> {
        #[allow(unused_assignments)]
        // needed to keep the lifetimes valid for this scope
        let mut groups_owned = None;

        let groups = if let Some((offset, len)) = slice {
            groups_owned = Some(self.get_groups().slice(offset, len));
            groups_owned.as_deref().unwrap()
        } else {
            self.get_groups()
        };

        by.iter()
            .map(|s| {
                match groups {
                    GroupsProxy::Idx(groups) => {
                        // SAFETY: groups are always in bounds.
                        let mut out = unsafe { s.take_slice_unchecked(groups.first()) };
                        if groups.is_sorted_flag() {
                            out.set_sorted_flag(s.is_sorted_flag());
                        };
                        out
                    }
                    GroupsProxy::Slice { groups, rolling } => {
                        if *rolling && !groups.is_empty() {
                            // Groups can be sliced.
                            let offset = groups[0][0];
                            let [upper_offset, upper_len] = groups[groups.len() - 1];
                            return s.slice(
                                offset as i64,
                                ((upper_offset + upper_len) - offset) as usize,
                            );
                        }

                        let indices = groups.iter().map(|&[first, _len]| first).collect_ca("");
                        // SAFETY: groups are always in bounds.
                        let mut out = unsafe { s.take_unchecked(&indices) };
                        // Sliced groups are always in order of discovery.
                        out.set_sorted_flag(s.is_sorted_flag());
                        out
                    }
                }
            })
            .collect()
    }

    fn keys_seq(&self, by: Vec<Series>) -> Vec<Series> {
        self.keys_sliced_seq(None, by)
    }

    fn prepare_agg_seq(
        &self,
        by: Vec<Series>,
        selected_agg: Option<Vec<String>>,
    ) -> PolarsResult<(Vec<Series>, Vec<Series>)> {
        let selection = match &selected_agg {
            Some(selection) => selection.clone(),
            None => {
                let by: Vec<_> = by.iter().map(|s| s.name()).collect();
                self.df
                    .get_column_names()
                    .into_iter()
                    .filter(|a| !by.contains(a))
                    .map(|s| s.to_string())
                    .collect()
            }
        };

        let keys = self.keys_seq(by);
        let agg_col = self.df.select_series(selection)?;
        Ok((keys, agg_col))
    }

    fn sum_seq(
        &self,
        by: Vec<Series>,
        selected_agg: Option<Vec<String>>,
    ) -> PolarsResult<DataFrame> {
        let (mut cols, agg_cols) = self.prepare_agg_seq(by, selected_agg)?;

        for agg_col in agg_cols {
            let new_name = fmt_group_by_column(agg_col.name(), GroupByMethod::Sum);
            let agg = unsafe { agg_col.agg_list(self.get_groups()) };
            let sum: ChunkedArray<Float32Type> = agg
                .list()?
                .into_iter()
                .map(|x| {
                    let arr = x.as_ref().unwrap().f32().unwrap();
                    arr.sum()
                })
                .collect_ca(&new_name);
            cols.push(sum.into_series());
        }
        DataFrame::new(cols)
    }
}

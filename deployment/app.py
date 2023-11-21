import streamlit as st
import pandas as pd
import numpy as np
import json
import pickle
import joblib

with open('model_xgb_tuned.pkl', 'rb') as file_1:
    model = joblib.load(file_1)

with open('model_transformer.pkl', 'rb') as file_2:
    transformer = joblib.load(file_2)

with st.form(key='form_credit_default'):
    ID = st.text_input('User ID', value='')
    bic = st.slider('Did the user click the basket icon', 0, 1, 1)
    bal = st.slider('Basket add list', 0, 1, 1)
    bad = st.slider('Basket add detail', 0, 1, 1)
    sb = st.slider('sort by', 0, 1, 1)
    ip = st.slider('image picker', 0, 1, 1)
    apc = st.slider('account page click', 0, 1, 1)
    pbc = st.slider('promo banner click', 0, 1, 1)
    dwa = st.slider('detail wishlist add', 0, 1, 1)
    lsd = st.slider('list size dropdown', 0, 1, 1)
    cmc = st.slider('closed minibasket click', 0, 1, 1)
    cdd = st.slider('checked delivery detail', 0, 1, 1)
    crd = st.slider('checked return detail', 0, 1, 1)
    si = st.slider('sign in', 0, 1, 1)
    sc = st.slider('saw checkout', 0, 1, 1)
    ss = st.slider('saw sizechart', 0, 1, 1)
    sd = st.slider('saw delivery', 0, 1, 1)
    sau = st.slider('saw account upgrade', 0, 1, 1)
    sh = st.slider('saw homepage', 0, 1, 1)
    dm = st.slider('device mobile', 0, 1, 1)
    dc = st.slider('device computer', 0, 1, 1)
    dt = st.slider('device tablet', 0, 1, 1)
    ru = st.slider('returning user', 0, 1, 1)
    lu = st.slider('loc uk', 0, 1, 1)
    o = st.slider('ordered', 0, 1, 1)

    submitted = st.form_submit_button('Predict')

test_data = pd.DataFrame({
  'UserID': [ID],
  'basket_icon_click': [bic],
  'basket_add_list': [bal],
  'basket_add_detail': [bad],
  'sort_by': [sb],
  'image_picker': [ip],
  'account_page_click': [apc],
  'promo_banner_click': [pbc],
  'detail_wishlist_add': [dwa],
  'list_size_dropdown': [lsd],
  'closed_minibasket_click': [cmc],
  'checked_delivery_detail': [cdd],
  'checked_returns_detail': [crd],
  'sign_in': [si],
  'saw_checkout': [sc],
  'saw_sizecharts': [ss],
  'saw_delivery': [sd],
  'saw_account_upgrade': [sau],
  'saw_homepage': [sh],
  'device_mobile': [dm],
  'device_computer': [dc],
  'device_tablet': [dt],
  'returning_user': [ru],
  'loc_uk': [lu],
  'ordered': [o]
  }
)

if submitted:
  y_pred_inf = model.predict(test_data)
  st.write('# Is the customer proceed to checkout : ', str(int(y_pred_inf)))
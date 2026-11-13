
import streamlit as st
import pandas as pd
import numpy as np
import cloudpickle
import os
import model_functions

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="Hotel Booking Intelligence",
    page_icon="🏨",
    layout="wide"
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR, "hotel_cancellation_model.pkl"
)

DATA_PATH = os.path.join(
    BASE_DIR, "hotel_bookings.csv"
)

# ==========================================
# 2. LOAD MODEL AND DATA
# ==========================================

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return cloudpickle.load(f)


@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)


try:
    model = load_model()
    df = load_data()

except Exception as e:
    st.error(f"Loading Error: {e}")
    st.stop()


# ==========================================
# 3. SIDEBAR NAVIGATION
# ==========================================

st.sidebar.title("🏨 Hotel Intelligence")

st.sidebar.caption("Booking Analytics & Prediction")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "Cancellation Prediction"
    ]
)

st.sidebar.divider()

st.sidebar.info(
    "Hotel Booking Intelligence System"
)


# ==========================================
# 4. DASHBOARD OVERVIEW
# ==========================================

if page == "Dashboard Overview":

    st.title("🏨 Hotel Booking Intelligence")
    st.write(
        "Monitor hotel booking performance, "
        "cancellation trends and business indicators."
    )

    
    st.divider()

    # ==========================================
    # DASHBOARD INTERACTIVE FILTERS
    # ==========================================

    st.subheader("Dashboard Filters")

    filter_col1, filter_col2, filter_col3 = st.columns(3)

    with filter_col1:
        selected_hotels = st.multiselect(
            "Hotel Type",
            options=sorted(df["hotel"].dropna().unique()),
            default=sorted(df["hotel"].dropna().unique())
        )

    with filter_col2:
        selected_years = st.multiselect(
            "Arrival Year",
            options=sorted(df["arrival_date_year"].dropna().unique()),
            default=sorted(df["arrival_date_year"].dropna().unique())
        )

    with filter_col3:
        selected_segments = st.multiselect(
            "Market Segment",
            options=sorted(df["market_segment"].dropna().unique()),
            default=sorted(df["market_segment"].dropna().unique())
        )

    # Apply Filters
    view = df[
        df["hotel"].isin(selected_hotels) &
        df["arrival_date_year"].isin(selected_years) &
        df["market_segment"].isin(selected_segments)
    ].copy()

    if view.empty:
        st.warning("No bookings match the selected filters.")
        st.stop()

    st.divider()

    # Dataset metrics
    total_bookings = len(view)


    cancelled = int(df["is_canceled"].sum())

    not_cancelled = total_bookings - cancelled

    cancellation_rate = (
        cancelled / total_bookings * 100
        if total_bookings else 0
    )

    active_rate = (
        not_cancelled / total_bookings * 100
        if total_bookings else 0
    )

    avg_adr = df["adr"].mean()

    # KPI CARDS
    st.subheader("Key Performance Indicators")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Total Bookings",
        f"{total_bookings:,}"
    )

    col2.metric(
        "Cancelled Bookings",
        f"{cancelled:,}"
    )

    col3.metric(
        "Cancellation Rate",
        f"{cancellation_rate:.2f}%"
    )

    col4.metric(
        "Average ADR",
        f"{avg_adr:,.2f}"
    )

    st.divider()

    # SECOND ROW
    st.subheader("Booking Overview")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Booking Status Distribution")

        status_data = pd.DataFrame({
            "Booking Status": [
                "Not Cancelled",
                "Cancelled"
            ],
            "Bookings": [
                not_cancelled,
                cancelled
            ]
        }).set_index("Booking Status")

        st.bar_chart(status_data)

        st.caption(
            f"Successful bookings: {active_rate:.2f}%"
        )

    with col2:

        st.markdown("#### Bookings by Hotel Type")

        hotel_counts = (
            df["hotel"]
            .value_counts()
            .rename("Bookings")
        )

        st.bar_chart(hotel_counts)

    st.divider()

    # CANCELLATION ANALYSIS
    st.subheader("Cancellation Analysis")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("#### Cancellation by Hotel Type")

        hotel_cancellation = (
            df.groupby("hotel")["is_canceled"]
            .mean()
            .mul(100)
            .rename("Cancellation Rate (%)")
        )

        st.bar_chart(hotel_cancellation)

    with col2:

        st.markdown("#### Cancellation by Deposit Type")

        deposit_cancellation = (
            df.groupby("deposit_type")["is_canceled"]
            .mean()
            .mul(100)
            .rename("Cancellation Rate (%)")
            .sort_values(ascending=False)
        )

        st.bar_chart(deposit_cancellation)

    st.divider()

    # MONTHLY ANALYSIS
    st.subheader("Monthly Booking Trends")

    month_order = [
        "January", "February", "March",
        "April", "May", "June",
        "July", "August", "September",
        "October", "November", "December"
    ]

    monthly_data = (
        df.groupby("arrival_date_month")["is_canceled"]
        .agg(["count", "mean"])
        .reindex(month_order)
    )

    monthly_data["mean"] = monthly_data["mean"] * 100

    col1, col2 = st.columns(2)

    with col1:

              
      monthly_data = (
          view.groupby("arrival_date_month")["is_canceled"]
          .agg(["count", "mean"])
          .reindex(month_order)
      )

      monthly_data["mean"] = monthly_data["mean"] * 100

      col1, col2 = st.columns(2)

      with col1:
          st.markdown("#### Monthly Bookings")

          st.bar_chart(
              monthly_data["count"].dropna(),
              y_label="Total Bookings"
          )

      with col2:
          st.markdown("#### Monthly Cancellation Rate")

          st.line_chart(
              monthly_data["mean"].dropna(),
              y_label="Cancellation Rate (%)"
          )


    st.divider()

    # BUSINESS SUMMARY
    st.subheader("Business Insights")

    st.write(
        f"• The dataset contains **{total_bookings:,} bookings**."
    )

    st.write(
        f"• **{cancellation_rate:.2f}%** of bookings were cancelled."
    )

    st.write(
        f"• The average daily rate (ADR) is **{avg_adr:.2f}**."
    )

    highest_hotel = hotel_cancellation.idxmax()

    st.write(
        f"• **{highest_hotel}** has the higher cancellation rate."
    )

    st.caption(
        "Insights are calculated from the uploaded historical dataset."
    )


# ==========================================
# 5. CANCELLATION PREDICTION
# ==========================================

elif page == "Cancellation Prediction":

    st.title("🔮 Hotel Cancellation Prediction")

    st.write(
        "Enter booking information to predict "
        "the probability of cancellation."
    )

    st.divider()

    with st.form("booking_form"):

        st.subheader("Hotel & Arrival Details")

        col1, col2, col3 = st.columns(3)

        with col1:
            hotel = st.selectbox(
                "Hotel Type",
                ["City Hotel", "Resort Hotel"]
            )

            arrival_date_year = st.selectbox(
                "Arrival Year",
                [2015, 2016, 2017]
            )

        with col2:
            arrival_date_month = st.selectbox(
                "Arrival Month",
                [
                    "January", "February", "March",
                    "April", "May", "June", "July",
                    "August", "September", "October",
                    "November", "December"
                ]
            )

            arrival_date_week_number = st.number_input(
                "Arrival Week Number", 1, 53, 25
            )

        with col3:
            arrival_date_day_of_month = st.number_input(
                "Arrival Day", 1, 31, 15
            )

            country = st.text_input(
                "Country Code", "PRT"
            )

        st.divider()

        st.subheader("Guest Information")

        col1, col2, col3 = st.columns(3)

        with col1:
            adults = st.number_input(
                "Adults", 0, 55, 2
            )

        with col2:
            children = st.number_input(
                "Children", 0, 10, 0
            )

        with col3:
            babies = st.number_input(
                "Babies", 0, 10, 0
            )

        st.divider()

        st.subheader("Stay Information")

        col1, col2 = st.columns(2)

        with col1:
            stays_in_weekend_nights = st.number_input(
                "Weekend Nights", 0, 19, 1
            )

            stays_in_week_nights = st.number_input(
                "Week Nights", 0, 50, 2
            )

            lead_time = st.number_input(
                "Lead Time (Days)", 0, 737, 30
            )

            meal = st.selectbox(
                "Meal",
                ["BB", "HB", "SC", "FB", "Undefined"]
            )

        with col2:
            reserved_room_type = st.text_input(
                "Reserved Room Type", "A"
            )

            assigned_room_type = st.text_input(
                "Assigned Room Type", "A"
            )

            adr = st.number_input(
                "Average Daily Rate",
                min_value=0.0,
                value=100.0
            )

        st.divider()

        st.subheader("Booking & Customer Details")

        col1, col2 = st.columns(2)

        with col1:
            market_segment = st.selectbox(
                "Market Segment",
                [
                    "Online TA", "Offline TA/TO",
                    "Groups", "Direct", "Corporate",
                    "Complementary", "Aviation",
                    "Undefined"
                ]
            )

            distribution_channel = st.selectbox(
                "Distribution Channel",
                [
                    "TA/TO", "Direct",
                    "Corporate", "GDS", "Undefined"
                ]
            )

            deposit_type = st.selectbox(
                "Deposit Type",
                [
                    "No Deposit",
                    "Non Refund",
                    "Refundable"
                ]
            )

            customer_type = st.selectbox(
                "Customer Type",
                [
                    "Transient",
                    "Transient-Party",
                    "Contract",
                    "Group"
                ]
            )

            agent = st.text_input("Agent ID", "9.0")

        with col2:
            is_repeated_guest = st.selectbox(
                "Repeated Guest", [0, 1]
            )

            previous_cancellations = st.number_input(
                "Previous Cancellations", 0, 100, 0
            )

            previous_bookings_not_canceled = st.number_input(
                "Previous Successful Bookings", 0, 100, 0
            )

            booking_changes = st.number_input(
                "Booking Changes", 0, 30, 0
            )

            days_in_waiting_list = st.number_input(
                "Days in Waiting List", 0, 400, 0
            )

            required_car_parking_spaces = st.number_input(
                "Required Parking Spaces", 0, 8, 0
            )

            total_of_special_requests = st.number_input(
                "Special Requests", 0, 5, 0
            )

        submit = st.form_submit_button(
            "Predict Cancellation",
            use_container_width=True
        )

    # ======================================
    # 6. PREDICTION RESULTS
    # ======================================

    if submit:

        if adults + children + babies == 0:

            st.error(
                "Please enter at least one guest."
            )

        else:

            input_data = pd.DataFrame([{
                "hotel": hotel,
                "lead_time": lead_time,
                "arrival_date_year": arrival_date_year,
                "arrival_date_month": arrival_date_month,
                "arrival_date_week_number": arrival_date_week_number,
                "arrival_date_day_of_month": arrival_date_day_of_month,
                "stays_in_weekend_nights": stays_in_weekend_nights,
                "stays_in_week_nights": stays_in_week_nights,
                "adults": adults,
                "children": children,
                "babies": babies,
                "meal": meal,
                "country": country.strip().upper(),
                "market_segment": market_segment,
                "distribution_channel": distribution_channel,
                "is_repeated_guest": is_repeated_guest,
                "previous_cancellations": previous_cancellations,
                "previous_bookings_not_canceled": previous_bookings_not_canceled,
                "reserved_room_type": reserved_room_type,
                "assigned_room_type": assigned_room_type,
                "booking_changes": booking_changes,
                "deposit_type": deposit_type,
                "agent": agent.strip(),
                "customer_type": customer_type,
                "adr": adr,
                "days_in_waiting_list": days_in_waiting_list,
                "required_car_parking_spaces": required_car_parking_spaces,
                "total_of_special_requests": total_of_special_requests
            }])

            try:

                prediction = model.predict(input_data)[0]

                probability = model.predict_proba(
                    input_data
                )[0][1]

                st.divider()

                st.header("Prediction Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Cancellation Probability",
                        f"{probability * 100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Predicted Status",
                        "Cancelled" if prediction == 1
                        else "Not Cancelled"
                    )

                st.progress(float(probability))

                if probability < 0.30:
                    st.success("Low Cancellation Risk")

                elif probability < 0.60:
                    st.warning("Medium Cancellation Risk")

                else:
                    st.error("High Cancellation Risk")

                st.caption(
                    "Predictions are estimates based on historical "
                    "booking data and do not guarantee actual outcomes."
                )

            except Exception as e:

                st.error(f"Prediction Error: {e}")


# ==========================================
# 7. FOOTER
# ==========================================

st.sidebar.divider()

st.sidebar.caption(
    "Hotel Booking Intelligence | Machine Learning Project"
)
